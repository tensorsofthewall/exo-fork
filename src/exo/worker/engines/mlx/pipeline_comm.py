"""Communication seam for pipeline-parallel execution.

Pipeline stages only need point-to-point activation transfer plus a few
cross-rank agreements. Tensor parallelism is deliberately NOT routed through
this protocol: it keeps using ``mx.distributed.Group`` directly.
"""

from collections.abc import Sequence
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

    def flush_sends(self, sends: Sequence[tuple[mx.array, int]]) -> None:
        """Send every queued (array, dst) pair, in order."""
        ...

    def step_complete(self) -> None:
        """The stage output of this forward step has been evaluated, so every array returned by ``recv_like`` for the step is dead.

        An implementation may recycle the receive destinations; one that does not has nothing to do.
        """
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

    def flush_sends(self, sends: Sequence[tuple[mx.array, int]]) -> None:
        for array, dst in sends:
            sent = self.send(array, dst)
            mx.async_eval(sent)

    def step_complete(self) -> None:
        pass

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


CommGroup = mx.distributed.Group | PipelineComm
"""What the runner threads through generation: an mx.distributed group (ring, JACCL) or a PipelineComm (e.g. TBCCL)."""


def as_pipeline_comm(comm: CommGroup) -> PipelineComm:
    if isinstance(comm, mx.distributed.Group):
        return MlxPipelineComm(comm)
    return comm


def comm_all_gather(array: mx.array, group: CommGroup | None) -> mx.array:
    if group is None or isinstance(group, mx.distributed.Group):
        return mx.distributed.all_gather(array, group=group)
    return group.all_gather(array)


def release_comm(group: CommGroup | None) -> None:
    """Close a PipelineComm (idempotent); mx.distributed groups have no explicit close."""
    if (
        group is not None
        and not isinstance(group, mx.distributed.Group)
        and hasattr(group, "close")
    ):
        group.close()
