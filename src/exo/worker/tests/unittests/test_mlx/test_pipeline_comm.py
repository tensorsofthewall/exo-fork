import mlx.core as mx

from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
)
from exo.worker.engines.mlx.pipeline_comm import MlxPipelineComm, as_pipeline_comm


class RecordingComm:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size
        self.calls: list[tuple[str, int]] = []

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def send(self, array: mx.array, dst: int) -> mx.array:
        self.calls.append(("send", dst))
        return array

    def recv_like(self, template: mx.array, src: int) -> mx.array:
        self.calls.append(("recv_like", src))
        return template + 1

    def all_gather(self, array: mx.array) -> mx.array:
        self.calls.append(("all_gather", -1))
        return mx.concatenate([array] * self._size, axis=0)

    def barrier(self) -> None:
        self.calls.append(("barrier", -1))

    def any_true(self, value: bool) -> bool:
        return value

    def close(self) -> None:
        pass


class Identity:
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x


def test_as_pipeline_comm_wraps_group_and_passes_through() -> None:
    group = mx.distributed.init()
    wrapped = as_pipeline_comm(group)
    assert isinstance(wrapped, MlxPipelineComm)
    assert (wrapped.rank(), wrapped.size()) == (group.rank(), group.size())
    rec = RecordingComm(0, 1)
    assert as_pipeline_comm(rec) is rec


def test_layers_use_the_comm_for_recv_send_all_gather() -> None:
    comm = RecordingComm(rank=1, size=3)
    first = PipelineFirstLayer(Identity(), r=1, group=comm)
    last = PipelineLastLayer(Identity(), r=1, s=3, group=comm)
    out = last(first(mx.zeros((1, 2))))
    assert [c[0] for c in comm.calls] == ["recv_like", "send", "all_gather"]
    assert comm.calls[0][1] == 0 and comm.calls[1][1] == 2
    assert out.shape == (1, 2)


def test_queued_sends_are_unsubmitted_until_flush_and_clearable() -> None:
    comm = RecordingComm(rank=0, size=2)
    last = PipelineLastLayer(Identity(), r=0, s=2, group=comm)
    last.queue_sends = True
    last.is_prefill = True
    last(mx.ones((1, 2)))
    assert comm.calls == []
    clear_prefill_sends()
    flush_prefill_sends()
    assert comm.calls == []
    last(mx.ones((1, 2)))
    flush_prefill_sends()
    assert comm.calls == [("send", 1)]
