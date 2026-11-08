# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix tokens lost to evicted recurrent-state snapshots, in the V2 iteration stats."""

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor import kv_cache_stats
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)


class _FakeKVCache:
    def __init__(self, divergence: int, hybrid_depth: int, reused: int):
        self._divergence = divergence
        self._hybrid_depth = hybrid_depth
        self.num_committed_tokens = reused

    def _get_num_reusable_tokens_before_pruning(self) -> int:
        return self._divergence

    def _get_num_reusable_tokens_before_hybrid_pruning(self) -> int:
        return self._hybrid_depth


def _manager(rank: int = 0) -> MambaHybridCacheManagerV2:
    # Only the counters the loss accounting touches; no cache pools.
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 1
    manager.is_estimating_kv_cache = False
    manager.mapping = SimpleNamespace(rank=rank)
    manager._snapshot_pruned_tokens_total = 0
    manager._page_pruned_tokens_total = 0
    manager._snapshot_matched_tokens_total = 0
    manager._iter_snapshot_pruned_tokens = 0
    manager._iter_snapshot_matched_tokens = 0
    manager._snapshot_loss_warned = False
    return manager


def test_counts_snapshot_and_page_loss_apart():
    manager = _manager()
    # Attention matched 20k of a 24k shared prefix, the deepest snapshot was at 4k
    manager._record_prefix_loss(_FakeKVCache(divergence=24_000, hybrid_depth=20_000, reused=4_000))
    manager._record_prefix_loss(_FakeKVCache(divergence=8_000, hybrid_depth=8_000, reused=8_000))
    assert manager._iter_snapshot_matched_tokens == 28_000
    assert manager._iter_snapshot_pruned_tokens == 16_000
    assert manager._page_pruned_tokens_total == 4_000


def test_estimation_dry_run_is_not_counted():
    manager = _manager()
    manager.is_estimating_kv_cache = True
    manager._record_prefix_loss(_FakeKVCache(divergence=100, hybrid_depth=100, reused=0))
    assert manager._snapshot_matched_tokens_total == 0


def test_warns_once_above_threshold(monkeypatch):
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    manager = _manager()
    matched = mamba_cache_manager_v2._SNAPSHOT_LOSS_MIN_MATCHED_TOKENS
    manager._record_prefix_loss(_FakeKVCache(matched, matched, reused=matched // 2))
    manager._warn_on_snapshot_loss()
    manager._warn_on_snapshot_loss()
    assert len(warnings) == 1 and "50%" in warnings[0]


def test_no_warning_below_threshold_or_on_other_ranks(monkeypatch):
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    matched = mamba_cache_manager_v2._SNAPSHOT_LOSS_MIN_MATCHED_TOKENS
    small_loss = _manager()
    small_loss._record_prefix_loss(_FakeKVCache(matched, matched, reused=matched - 1))
    small_loss._warn_on_snapshot_loss()
    other_rank = _manager(rank=1)
    other_rank._record_prefix_loss(_FakeKVCache(matched, matched, reused=0))
    other_rank._warn_on_snapshot_loss()
    assert warnings == []


def test_serialized_only_for_recurrent_models():
    report = kv_cache_stats.KVCacheV2IterationStatsReport(by_window_size={}, by_pool_group={})
    stats = {}
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert "iterSnapshotPrunedTokens" not in stats
    report.snapshot_matched_tokens = 28_000
    report.snapshot_pruned_tokens = 16_000
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert stats["iterSnapshotMatchedTokens"] == 28_000
    assert stats["iterSnapshotPrunedTokens"] == 16_000
