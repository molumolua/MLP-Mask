# Local development environment

## Python

- Use the existing `molu` Conda environment for every Python command in this repository.
- The interpreter is `/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python`.
- Do not rely on shell activation or an unqualified `python`, `pip`, or `pytest`; Codex shells are non-interactive and may otherwise select the base environment.
- Run Python tools as `/opt/homebrew/Caskroom/miniconda/base/envs/molu/bin/python -m <module>`.
- If dependencies must be reinstalled, use `--index-url https://pypi.org/simple`; the machine's configured Tsinghua mirror currently fails TLS negotiation. Do not change the user's global pip configuration.

## Verification

- Run `./scripts/check-local-env` to verify imports and a local CPU Ray task.
- Ray worker startup may require running that script outside the macOS process/network sandbox.
- Run `./scripts/test-local` for the repository test suite. Extra pytest arguments may be appended, for example `./scripts/test-local recipe/mlp_channel_mask -q`.
- `requirements-test-local.txt` documents the minimum local test dependencies.
- This machine is Apple Silicon macOS without CUDA. Unit tests and Ray CPU smoke tests are supported locally; full verl/vLLM training still requires a suitable Linux CUDA host, model files, and datasets.
