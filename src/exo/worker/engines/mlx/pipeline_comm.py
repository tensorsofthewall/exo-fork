"""Communication seam for pipeline-parallel execution.

Pipeline stages only need point-to-point activation transfer plus a few
cross-rank agreements. Tensor parallelism is deliberately NOT routed through
this protocol: it keeps using ``mx.distributed.Group`` directly.
"""

from typing import Protocol

import mlx.core as mx


class PipelineComm(Protocol):
    def rank(self) -> int: ...

    def size(self) -> int: ...

    def send(self, array: mx.array, dst: int) -> mx.array:
        """Send an already-evaluated array; returns an array to depend on."""
        ...

    def recv_like(self, template: mx.array, src: int) -> mx.array: ...

    def all_gather(self, array: mx.array) -> mx.array:
        """Rank-order concatenation along axis 0."""
        ...

    def barrier(self) -> None: ...

    def any_true(self, value: bool) -> bool: ...

    def close(self) -> None: ...


class MlxPipelineComm:
    """Thin wrapper over ``mx.distributed.Group``; behaviour-preserving."""

    def __init__(self, group: mx.distributed.Group):
        self.group: mx.distributed.Group = group

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def send(self, array: mx.array, dst: int) -> mx.array:
        return mx.distributed.send(array, dst, group=self.group)

    def recv_like(self, template: mx.array, src: int) -> mx.array:
        return mx.distributed.recv_like(template, src, group=self.group)

    def all_gather(self, array: mx.array) -> mx.array:
        return mx.distributed.all_gather(array, group=self.group)

    def barrier(self) -> None:
        mx.eval(
            mx.distributed.all_sum(
                mx.array(1.0),
                group=self.group,
                stream=mx.default_stream(mx.Device(mx.cpu)),
            )
        )

    def any_true(self, value: bool) -> bool:
        num_true = mx.distributed.all_sum(
            mx.array(value),
            group=self.group,
            stream=mx.default_stream(mx.Device(mx.cpu)),
        )
        mx.eval(num_true)
        return num_true.item() > 0

    def close(self) -> None:
        pass


def as_pipeline_comm(comm: "PipelineComm | mx.distributed.Group") -> PipelineComm:
    if isinstance(comm, mx.distributed.Group):
        return MlxPipelineComm(comm)
    return comm
