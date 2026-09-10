"""HF and vLLM dense SwiGLU adapters; buffers stay at fixed addresses."""

import re
from types import MethodType

import torch

from .intervention import MLPChannelRDropController

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.mlp(?:\.|$)")

def install_hf_mlp_intervention(
    model: torch.nn.Module, controller: MLPChannelRDropController
) -> list[str]:
    """Patch dense HF Qwen/Llama-style SwiGLU MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_rdrop_patched", False):
            if getattr(module, "_mlp_channel_rdrop_controller", None) is not controller:
                raise RuntimeError(f"HF MLP {name} was patched with a different controller")
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        def forward(this, hidden_state, *args, _layer_idx=layer_idx, **kwargs):
            if args or kwargs:
                raise TypeError("patched dense MLP expects only hidden_state")
            activation = this.act_fn(this.gate_proj(hidden_state)) * this.up_proj(hidden_state)
            activation = controller.apply(_layer_idx, activation)
            return this.down_proj(activation)

        module.forward = MethodType(forward, module)
        module._mlp_channel_rdrop_patched = True
        module._mlp_channel_rdrop_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="HF actor")
    return patched


_VLLM_ACTIVE_MASK_BUFFER = "_mlp_channel_rdrop_gain"


def _install_vllm_active_buffer(
    module: torch.nn.Module,
    controller: MLPChannelRDropController,
    layer_idx: int,
) -> torch.Tensor:
    existing = module._buffers.get(_VLLM_ACTIVE_MASK_BUFFER)
    if existing is None:
        if hasattr(module, _VLLM_ACTIVE_MASK_BUFFER):
            raise RuntimeError(
                f"vLLM MLP attribute {_VLLM_ACTIVE_MASK_BUFFER!r} exists but is not a buffer"
            )
        reference = getattr(getattr(module, "down_proj", None), "weight", None)
        if reference is None:
            reference = next(module.parameters(), None)
        if reference is None:
            raise RuntimeError("cannot allocate vLLM gain: MLP has no parameter")
        local_width = controller.intermediate_size // controller.tp_size
        device = None if reference.device.type == "meta" else reference.device
        existing = torch.ones(local_width, device=device, dtype=reference.dtype)
        module.register_buffer(_VLLM_ACTIVE_MASK_BUFFER, existing, persistent=False)
    return controller.register_active_buffer(layer_idx, existing)


def install_vllm_mlp_intervention(
    model: torch.nn.Module, controller: MLPChannelRDropController
) -> list[str]:
    """Patch already-built dense vLLM Qwen/Llama-style MLP instances."""
    patched: list[str] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _layer_index(name)
        if layer_idx is None or layer_idx in seen_layers:
            continue
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "act_fn")):
            continue
        if getattr(module, "_mlp_channel_rdrop_patched", False):
            if getattr(module, "_mlp_channel_rdrop_controller", None) is not controller:
                raise RuntimeError(f"vLLM MLP {name} was patched with a different controller")
            _install_vllm_active_buffer(module, controller, layer_idx)
            patched.append(name)
            seen_layers.add(layer_idx)
            continue

        _install_vllm_active_buffer(module, controller, layer_idx)

        def forward(this, hidden_state, *args, **kwargs):
            if args or kwargs:
                raise TypeError("patched vLLM dense MLP expects only hidden_state")
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_MASK_BUFFER)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        module.forward = MethodType(forward, module)
        module._mlp_channel_rdrop_patched = True
        module._mlp_channel_rdrop_controller = controller
        patched.append(name)
        seen_layers.add(layer_idx)

    _validate_patched_layers(controller, seen_layers, backend="vLLM rollout")
    return patched


def install_vllm_class_intervention(
    controller: MLPChannelRDropController,
) -> list[str]:
    """Patch vLLM Qwen MLP classes before model build and CUDA-graph capture."""
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    classes = [Qwen2MLP]
    try:
        from vllm.model_executor.models.qwen3 import Qwen3MLP

        if Qwen3MLP not in classes:
            classes.append(Qwen3MLP)
    except ImportError:
        pass

    patched_names: list[str] = []
    for cls in classes:
        existing_controller = getattr(cls, "_mlp_channel_rdrop_controller", None)
        if existing_controller is not None:
            if existing_controller is not controller:
                raise RuntimeError(f"{cls.__name__} is already patched with another controller")
            patched_names.append(cls.__name__)
            continue

        original_init = cls.__init__

        def patched_init(this, *args, __original_init=original_init, **kwargs):
            __original_init(this, *args, **kwargs)
            prefix = kwargs.get("prefix", args[4] if len(args) > 4 else "")
            layer_idx = _layer_index(str(prefix))
            if layer_idx is None:
                raise RuntimeError(f"cannot infer vLLM MLP layer from prefix {prefix!r}")
            this._mlp_channel_rdrop_layer_idx = layer_idx
            this._mlp_channel_rdrop_patched = True
            this._mlp_channel_rdrop_controller = controller
            _install_vllm_active_buffer(this, controller, layer_idx)

        def patched_forward(this, hidden_state):
            gate_up = this.gate_up_proj(hidden_state)
            gate_up = gate_up[0] if isinstance(gate_up, tuple) else gate_up
            activation = this.act_fn(gate_up)
            activation = activation * getattr(this, _VLLM_ACTIVE_MASK_BUFFER)
            output = this.down_proj(activation)
            return output[0] if isinstance(output, tuple) else output

        cls.__init__ = patched_init
        cls.forward = patched_forward
        cls._mlp_channel_rdrop_controller = controller
        patched_names.append(cls.__name__)
    return patched_names


def _layer_index(module_name: str) -> int | None:
    normalized_name = ".".join(
        part for part in module_name.split(".") if part != "_fsdp_wrapped_module"
    )
    match = _LAYER_RE.search(normalized_name)
    return int(match.group(1)) if match else None


def _validate_patched_layers(
    controller: MLPChannelRDropController,
    seen_layers: set[int],
    *,
    backend: str,
) -> None:
    expected = set(range(controller.num_layers))
    if seen_layers != expected:
        missing = sorted(expected - seen_layers)
        extra = sorted(seen_layers - expected)
        raise RuntimeError(
            f"{backend}: expected dense MLPs for {controller.num_layers} layers; "
            f"missing={missing}, extra={extra}. MoE and unknown layouts are unsupported."
        )
