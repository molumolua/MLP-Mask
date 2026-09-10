import pytest
import torch

from .kl import build_distribution, detached_log_probs, select_response_logits, symmetric_kl_sum


def reference_kl(a, b):
    loga, logb = a.log_softmax(-1), b.log_softmax(-1)
    return 0.5 * ((loga.exp() - logb.exp()) * (loga - logb)).sum()


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_exact_scalar_and_both_partial_gradients(chunk_size):
    torch.manual_seed(8)
    a = torch.randn(7, 13, dtype=torch.double, requires_grad=True)
    b = torch.randn(7, 13, dtype=torch.double, requires_grad=True)
    expected = reference_kl(a, b)
    da, db = torch.autograd.grad(expected, (a, b))
    partial_a = symmetric_kl_sum(a, detached_log_probs(b), chunk_size=chunk_size)
    partial_b = symmetric_kl_sum(b, detached_log_probs(a), chunk_size=chunk_size)
    torch.testing.assert_close(partial_a, expected)
    torch.testing.assert_close(partial_b, expected)
    torch.testing.assert_close(torch.autograd.grad(partial_a, a)[0], da)
    torch.testing.assert_close(torch.autograd.grad(partial_b, b)[0], db)


def test_gradcheck_and_zero_for_identical_distributions():
    torch.manual_seed(9)
    a = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
    b = detached_log_probs(torch.randn_like(a))
    assert torch.autograd.gradcheck(lambda x: symmetric_kl_sum(x, b, chunk_size=1), (a,))
    loss = symmetric_kl_sum(a, detached_log_probs(a))
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(a.grad, torch.zeros_like(a))


def test_padding_removal_empty_masks_and_extreme_logits():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    torch.testing.assert_close(select_response_logits(logits, mask),
                               select_response_logits(logits.reshape(6, 5), mask.reshape(1, 6)))
    selected = select_response_logits(logits, mask * 0)
    loss = symmetric_kl_sum(selected, detached_log_probs(selected))
    loss.backward()
    assert loss == 0
    assert logits.grad.abs().sum() == 0
    a = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
    b = torch.tensor([[-1000., 1000., 0.]])
    loss = symmetric_kl_sum(a, detached_log_probs(b))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()


def coarsened_reference(a, b, ids):
    def compact(x):
        top = x.gather(-1, ids)
        tail = x.scatter(-1, ids, -torch.inf).logsumexp(-1, keepdim=True)
        return torch.cat((top, tail), -1)
    return reference_kl(compact(a), compact(b))


@pytest.mark.parametrize("top_k", [1, 4, 12])
def test_topk_tail_scalar_two_sided_gradient_and_data_processing_bound(top_k):
    torch.manual_seed(6)
    a = torch.randn(5, 13, dtype=torch.double, requires_grad=True)
    b = torch.randn(5, 13, dtype=torch.double, requires_grad=True)
    da = build_distribution(a, top_k=top_k, chunk_size=2)
    db = build_distribution(b, top_k=0, token_ids=da.token_ids, chunk_size=3)
    expected = coarsened_reference(a, b, da.token_ids)
    ga, gb = torch.autograd.grad(expected, (a, b))
    la = symmetric_kl_sum(a, db.log_probs, token_ids=da.token_ids, chunk_size=2)
    lb = symmetric_kl_sum(b, da.log_probs, token_ids=da.token_ids, chunk_size=1)
    torch.testing.assert_close(la, expected)
    torch.testing.assert_close(lb, expected)
    torch.testing.assert_close(torch.autograd.grad(la, a)[0], ga)
    torch.testing.assert_close(torch.autograd.grad(lb, b)[0], gb)
    assert expected <= reference_kl(a, b) + 1e-12
    assert db.token_ids is da.token_ids
    assert torch.autograd.gradcheck(lambda x: symmetric_kl_sum(x, da.log_probs, token_ids=da.token_ids), (b,))


def test_topk_tail_handles_nearly_all_mass_on_one_token():
    a = torch.tensor([[1000., 0., -1000.]], requires_grad=True)
    b = torch.tensor([[-1000., 0., 1000.]])
    ids = torch.tensor([[0]])
    target = build_distribution(b, token_ids=ids)
    loss = symmetric_kl_sum(a, target.log_probs, token_ids=ids)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
