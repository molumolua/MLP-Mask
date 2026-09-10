import pytest
import torch

from .kl import build_distribution, detached_log_probs, forward_kl_sum, select_response_logits, slice_distribution_rows


def reference_kl(a, b):
    """Independent KL(teacher a || student b) reference; callers detach a."""
    loga, logb = a.log_softmax(-1), b.log_softmax(-1)
    return (loga.exp() * (loga - logb)).sum()


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_exact_scalar_student_gradient_and_detached_teacher(chunk_size):
    torch.manual_seed(8)
    a = torch.randn(7, 13, dtype=torch.double, requires_grad=True)
    b = torch.randn(7, 13, dtype=torch.double, requires_grad=True)
    expected = reference_kl(a.detach(), b)
    db = torch.autograd.grad(expected, b)[0]
    actual = forward_kl_sum(b, detached_log_probs(a), chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(b.grad, db)
    assert a.grad is None
    with pytest.raises(ValueError, match="detached"):
        forward_kl_sum(b, a.log_softmax(-1))


def test_gradcheck_and_zero_for_identical_distributions():
    torch.manual_seed(9)
    a = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
    b = detached_log_probs(torch.randn_like(a))
    assert torch.autograd.gradcheck(lambda x: forward_kl_sum(x, b, chunk_size=1), (a,))
    loss = forward_kl_sum(a, detached_log_probs(a))
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(a.grad, torch.zeros_like(a))


def test_padding_removal_empty_masks_and_extreme_logits():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    torch.testing.assert_close(select_response_logits(logits, mask),
                               select_response_logits(logits.reshape(6, 5), mask.reshape(1, 6)))
    selected = select_response_logits(logits, mask * 0)
    loss = forward_kl_sum(selected, detached_log_probs(selected))
    loss.backward()
    assert loss == 0
    assert logits.grad.abs().sum() == 0
    a = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
    b = torch.tensor([[-1000., 1000., 0.]])
    loss = forward_kl_sum(a, detached_log_probs(b))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()


def coarsened_reference(a, b, ids):
    def compact(x):
        top = x.gather(-1, ids)
        tail = x.scatter(-1, ids, -torch.inf).logsumexp(-1, keepdim=True)
        return torch.cat((top, tail), -1)
    return reference_kl(compact(a), compact(b))


@pytest.mark.parametrize("top_k", [1, 4, 12])
def test_topk_tail_student_gradient_and_data_processing_bound(top_k):
    torch.manual_seed(6)
    a = torch.randn(5, 13, dtype=torch.double, requires_grad=True)
    b = torch.randn(5, 13, dtype=torch.double, requires_grad=True)
    da = build_distribution(a, top_k=top_k, chunk_size=2)
    expected = coarsened_reference(a.detach(), b, da.token_ids)
    gb = torch.autograd.grad(expected, b)[0]
    lb = forward_kl_sum(b, da.log_probs, token_ids=da.token_ids, chunk_size=1)
    torch.testing.assert_close(lb, expected)
    lb.backward()
    torch.testing.assert_close(b.grad, gb)
    assert a.grad is None and not da.log_probs.requires_grad
    assert expected <= reference_kl(a, b) + 1e-12
    assert torch.autograd.gradcheck(lambda x: forward_kl_sum(x, da.log_probs, token_ids=da.token_ids), (b,))


def test_topk_tail_handles_nearly_all_mass_on_one_token():
    a = torch.tensor([[1000., 0., -1000.]], requires_grad=True)
    b = torch.tensor([[-1000., 0., 1000.]])
    ids = torch.tensor([[0]])
    target = build_distribution(b, token_ids=ids)
    loss = forward_kl_sum(a, target.log_probs, token_ids=ids)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()


@pytest.mark.parametrize("top_k", [0, 3])
def test_teacher_row_slices_follow_valid_tokens_including_empty_rows(top_k):
    logits = torch.randn(4, 3, 7, requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0]])
    teacher = build_distribution(select_response_logits(logits, mask), top_k=top_k)
    counts = tuple(mask.sum(-1).tolist())
    sliced = slice_distribution_rows(teacher, counts, 1, 3)
    torch.testing.assert_close(sliced.log_probs, teacher.log_probs[1:4])
    if top_k:
        torch.testing.assert_close(sliced.token_ids, teacher.token_ids[1:4])
    assert not sliced.log_probs.requires_grad
    empty = slice_distribution_rows(teacher, counts, 1, 2)
    assert empty.log_probs.shape[0] == 0
    student = select_response_logits(logits[1:2], mask[1:2])
    loss = forward_kl_sum(student, empty.log_probs, token_ids=empty.token_ids)
    loss.backward()
    assert loss == 0 and logits.grad.abs().sum() == 0
