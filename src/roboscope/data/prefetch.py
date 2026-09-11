"""Overlap pinned-memory host-to-device transfers with computation, preserving sample order."""

import torch


def device_batches(loader, enabled=True):
    if not enabled:
        for batch in loader:
            yield {k: v.to("cuda:0", non_blocking=True) for k, v in batch.items()}
        return
    stream = torch.cuda.Stream()
    iterator = iter(loader)

    def fetch():
        try:
            host = next(iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            return {k: v.to("cuda:0", non_blocking=True) for k, v in host.items()}

    next_batch = fetch()
    while next_batch is not None:
        torch.cuda.current_stream().wait_stream(stream)
        batch = next_batch
        # 告知 allocator：这些 tensor 会在计算流上使用，不能提前复用其显存。
        for tensor in batch.values():
            tensor.record_stream(torch.cuda.current_stream())
        next_batch = fetch()
        yield batch
