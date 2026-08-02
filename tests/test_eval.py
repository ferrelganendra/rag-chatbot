"""Tests for evaluation metrics."""

from eval.metrics import hit_rate, mrr


def test_hit_rate_perfect():
    assert hit_rate(["a.md"], ["a.md", "b.md"]) == 1.0


def test_hit_rate_miss():
    assert hit_rate(["a.md"], ["b.md", "c.md"]) == 0.0


def test_hit_rate_k_limit():
    # relevant doc at position 3, k=2 → missed
    assert hit_rate(["a.md"], ["x.md", "y.md", "a.md"], k=2) == 0.0
    # relevant doc at position 3, k=3 → hit
    assert hit_rate(["a.md"], ["x.md", "y.md", "a.md"], k=3) == 1.0


def test_mrr_first():
    assert mrr(["a.md"], ["a.md", "b.md"]) == 1.0


def test_mrr_third():
    assert abs(mrr(["a.md"], ["x.md", "y.md", "a.md"]) - 0.3333) < 0.01


def test_mrr_miss():
    assert mrr(["a.md"], ["x.md", "y.md"]) == 0.0


def test_hit_rate_empty():
    assert hit_rate([], ["a.md"]) == 0.0
