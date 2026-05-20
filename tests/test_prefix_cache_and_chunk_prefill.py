"""Tests verifying prefix cache and chunk prefill are effective."""

from python.core.block_pool import BlockPool
from python.core.scheduler import (
    Request,
    Scheduler,
    SchedulerConfig,
)


def _make_scheduler(block_size=4, num_blocks=32, max_tokens=8, threshold=4):
    block_pool = BlockPool(num_blocks=num_blocks, block_size=block_size)
    config = SchedulerConfig(
        max_num_running_reqs=8,
        max_num_scheduled_tokens=max_tokens,
        long_prefill_token_threshold=threshold,
        max_seq_len=64,
    )
    return Scheduler(config=config, block_pool=block_pool)


def test_chunk_prefill_splits_long_prompt():
    """A prompt longer than long_prefill_token_threshold is split across steps."""
    scheduler = _make_scheduler(block_size=4, max_tokens=16, threshold=4)
    prompt = list(range(12))  # 12 tokens, threshold=4 -> 3 chunks

    request = Request(request_id="r1", prompt_token_ids=prompt, max_new_tokens=1)
    scheduler.add_request(request)

    # Step 1: should schedule at most 4 tokens
    out1 = scheduler.schedule()
    assert len(out1.scheduled_requests) == 1
    sr1 = out1.scheduled_requests[0]
    assert sr1.num_new_tokens == 4
    assert sr1.num_computed_tokens == 0
    assert sr1.is_prefill is True

    # Simulate execution: advance num_computed_tokens
    scheduler.update_from_output(out1, {})

    # Step 2: next chunk
    out2 = scheduler.schedule()
    assert len(out2.scheduled_requests) == 1
    sr2 = out2.scheduled_requests[0]
    assert sr2.num_new_tokens == 4
    assert sr2.num_computed_tokens == 4
    assert sr2.is_prefill is True

    scheduler.update_from_output(out2, {})

    # Step 3: final chunk
    out3 = scheduler.schedule()
    assert len(out3.scheduled_requests) == 1
    sr3 = out3.scheduled_requests[0]
    assert sr3.num_new_tokens == 4
    assert sr3.num_computed_tokens == 8
    assert sr3.is_prefill is True

    scheduler.update_from_output(out3, {"r1": 99})


def test_prefix_cache_reduces_computed_tokens():
    """Second request with same prefix reuses cached blocks."""
    scheduler = _make_scheduler(block_size=4, max_tokens=32, threshold=0)
    prompt = list(range(16))  # 4 full blocks

    # First request: full prefill
    r1 = Request(request_id="r1", prompt_token_ids=prompt, max_new_tokens=1)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    sr1 = out1.scheduled_requests[0]
    assert sr1.num_computed_tokens == 0
    assert sr1.num_new_tokens == 16

    scheduler.update_from_output(out1, {"r1": 99})
    scheduler.finish_request("r1", status=r1.status)

    # Second request with same prefix: should hit cache
    r2 = Request(request_id="r2", prompt_token_ids=prompt + [100], max_new_tokens=1)
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    sr2 = out2.scheduled_requests[0]
    # All 4 blocks (16 tokens) should be cached
    assert sr2.num_computed_tokens == 16
    assert sr2.num_new_tokens == 1  # only the extra token


def test_block_ids_passed_to_scheduled_request():
    """ScheduledRequest contains the correct block_ids."""
    scheduler = _make_scheduler(block_size=4, max_tokens=32, threshold=0)
    prompt = list(range(8))  # 2 full blocks

    request = Request(request_id="r1", prompt_token_ids=prompt, max_new_tokens=1)
    scheduler.add_request(request)
    out = scheduler.schedule()
    sr = out.scheduled_requests[0]

    # Should have 2 blocks allocated
    assert len(sr.block_ids) == 2
    assert sr.block_ids == request.cached_block_ids + request.allocated_block_ids


def test_chunk_prefill_with_prefix_cache():
    """Chunk prefill works correctly after prefix cache hit."""
    scheduler = _make_scheduler(block_size=4, max_tokens=32, threshold=4)
    prefix = list(range(8))  # 2 full blocks

    # First request establishes cache
    r1 = Request(request_id="r1", prompt_token_ids=prefix, max_new_tokens=1)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    scheduler.update_from_output(out1, {})
    out1b = scheduler.schedule()
    scheduler.update_from_output(out1b, {"r1": 99})
    scheduler.finish_request("r1", status=r1.status)

    # Second request: prefix cached (8 tokens), plus 6 new tokens
    r2 = Request(
        request_id="r2",
        prompt_token_ids=prefix + list(range(100, 106)),
        max_new_tokens=1,
    )
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    sr2 = out2.scheduled_requests[0]
    # 8 tokens cached, 6 new tokens but threshold=4 -> first chunk is 4
    assert sr2.num_computed_tokens == 8
    assert sr2.num_new_tokens == 4
    assert len(sr2.block_ids) >= 3  # 2 cached + at least 1 new
