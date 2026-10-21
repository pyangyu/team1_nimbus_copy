import pytest

from budget_classifier import Buckets


def test_default_has_ten_literal_buckets():
    b = Buckets()
    assert b.n == 10
    assert b.labels() == ["0-32", "32-128", "128-256", "256-512", "512-1K", "1K-2K",
                          "2K-4K", "4K-8K", "8K-16K", "16K+"]


@pytest.mark.parametrize("tokens,idx", [(0, 0), (31, 0), (32, 1), (127, 1), (128, 2), (1023, 4),
                                        (1024, 5), (16383, 8), (16384, 9), (10**6, 9), (-5, 0)])
def test_index(tokens, idx):
    assert Buckets().index(tokens) == idx


def test_budget_is_upper_edge():
    b = Buckets()
    assert [b.budget(i) for i in range(b.n)] == [32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


@pytest.mark.parametrize("edges", [(0, 10), (0, 10, 5), (1, 10, 20), tuple(range(0, 120, 10))])
def test_rejects_bad_edges(edges):
    with pytest.raises(ValueError):
        Buckets(edges)


def test_eight_buckets_allowed():
    assert Buckets((0, 32, 128, 512, 1024, 2048, 4096, 8192, 32768)).n == 8
