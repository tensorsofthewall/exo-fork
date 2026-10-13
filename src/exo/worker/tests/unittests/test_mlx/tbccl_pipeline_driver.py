# type: ignore
"""Driver for the synthetic pipeline test (run as a subprocess by test_tbccl_pipeline.py; ranks are spawned so MLX is never forked).

Synthetic pipeline through exo's real pipeline path: unsharded vs MlxRing vs MlxTbccl.

A tiny deterministic model (no weights downloaded) is sharded by ``pipeline_auto_parallel``; every rank runs in its own process. The TBCCL
ranks bootstrap through the real RunnerByteExchange and ``init_tbccl_pipeline_comm`` (so the bootstrap blobs travel as opaque bytes), and
every activation crossing a stage boundary is hashed on the sender and on the receiver: because transport is byte-generic they must match.
Usage: ``python -m <this module> tbccl|ring <world> <dtype>`` prints one JSON line.
"""

import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import threading
import time
from collections.abc import Sequence
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.runner_bytes import RunnerBytesMessage
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxTbcclInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import MpReceiver, MpSender, mp_channel
from exo.worker.engines.mlx.auto_parallel import (
    flush_prefill_sends,
    pipeline_auto_parallel,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.pipeline_comm import (
    MlxPipelineComm,
    PipelineComm,
)
from exo.worker.engines.mlx.tbccl_comm import init_tbccl_pipeline_comm
from exo.worker.runner.byte_exchange import RunnerByteExchange
from exo.worker.runner_bytes_hub import RunnerBytesHub

DIM = 64
N_LAYERS = 8
CHUNKS = 3


class Block(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.w = mx.random.normal((DIM, DIM), key=mx.random.key(seed)) * 0.1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x + mx.tanh(
            x @ self.w.astype(x.dtype)
        )  # keep the activation dtype, like a real model


class Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [Block(i) for i in range(N_LAYERS)]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Inner()

    @property
    def layers(self) -> list[nn.Module]:  # mlx_lm models expose this too
        return self.model.layers

    def __call__(self, x: mx.array, cache: object = None) -> mx.array:
        for layer in self.model.layers:
            x = layer(x, cache=cache)
        return x


def _digest(a: mx.array) -> str:
    mx.eval(a)
    return hashlib.sha256(bytes(memoryview(a.view(mx.uint8)))).hexdigest()


class RecordingComm:
    """Hashes every activation that crosses a stage boundary, on the sending and on the receiving side."""

    def __init__(self, inner: PipelineComm) -> None:
        self.inner = inner
        self.sent: list[str] = []
        self.recv: list[str] = []

    def rank(self) -> int:
        return self.inner.rank()

    def size(self) -> int:
        return self.inner.size()

    def send(self, array: mx.array, dst: int) -> mx.array:
        self.sent.append(_digest(array))
        return self.inner.send(array, dst)

    def recv_like(self, template: mx.array, src: int) -> mx.array:
        out = self.inner.recv_like(template, src)
        self.recv.append(_digest(out))
        return out

    def all_gather(self, array: mx.array) -> mx.array:
        return self.inner.all_gather(array)

    def flush_sends(self, sends: Sequence[tuple[mx.array, int]]) -> None:
        for array, _ in sends:
            self.sent.append(_digest(array))
        self.inner.flush_sends(sends)

    def barrier(self) -> None:
        self.inner.barrier()

    def any_true(self, value: bool) -> bool:
        return self.inner.any_true(value)

    def close(self) -> None:
        self.inner.close()


def _inputs(dtype: mx.Dtype) -> list[mx.array]:
    return [
        (mx.random.normal((1, 8, DIM), key=mx.random.key(100 + i))).astype(dtype)
        for i in range(CHUNKS)
    ]


def _layout(world: int) -> list[tuple[int, int]]:
    base, extra = divmod(N_LAYERS, world)
    out: list[tuple[int, int]] = []
    start = 0
    for r in range(world):
        n = base + (1 if r < extra else 0)
        out.append((start, start + n))
        start += n
    return out


def _model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("synthetic/tiny"),
        storage_size=Memory.from_kb(1),
        n_layers=N_LAYERS,
        hidden_size=DIM,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    )


def _shard(rank: int, world: int) -> PipelineShardMetadata:
    start, end = _layout(world)[rank]
    return PipelineShardMetadata(
        model_card=_model_card(),
        device_rank=rank,
        world_size=world,
        start_layer=start,
        end_layer=end,
        n_layers=N_LAYERS,
    )


def _run_model(
    comm: PipelineComm, rank: int, world: int, dtype_name: str
) -> dict[str, Any]:
    dtype = getattr(mx, dtype_name)
    xs = _inputs(dtype)
    reference_model = TinyModel()
    refs = [reference_model(x) for x in xs]
    mx.eval(refs)

    model = TinyModel()
    gen = pipeline_auto_parallel(model, comm, _shard(rank, world))
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        model = cast(TinyModel, stop.value)

    result: dict[str, Any] = {}
    # decode-style: not prefill -> every rank takes part in the final all_gather and keeps the last rank's rows
    set_pipeline_prefill(model, is_prefill=False)
    decode_ok: list[bool] = []
    diffs: list[float] = []
    for x, ref in zip(xs, refs, strict=True):
        out = model(x)
        mx.eval(out)
        decode_ok.append(_digest(out) == _digest(ref))
        diffs.append(
            float(
                mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
            )
        )
    result["decode_matches_unsharded"] = all(decode_ok)
    result["decode_max_abs_diff"] = max(diffs)  # nonzero only across different devices

    # prefill-style: queued sends flushed after each chunk; the last rank holds the real output
    set_pipeline_prefill(model, is_prefill=True)
    set_pipeline_queue_sends(model, queue_sends=True)
    prefill_ok: list[bool] = []
    for x, ref in zip(xs, refs, strict=True):
        out = model(x)
        mx.eval(out)
        flush_prefill_sends()
        if rank == world - 1:
            prefill_ok.append(_digest(out) == _digest(ref))
    set_pipeline_queue_sends(model, queue_sends=False)
    result["prefill_matches_unsharded"] = all(prefill_ok) if rank == world - 1 else None
    comm.barrier()
    if isinstance(comm, RecordingComm):
        result["sent"] = comm.sent
        result["recv"] = comm.recv
    return result


def _tbccl_rank(
    rank: int,
    world: int,
    dtype_name: str,
    node_ids: list[str],
    instance_id: str,
    out_s: MpSender[RunnerBytesMessage],
    in_r: MpReceiver[RunnerBytesMessage],
    result_q: Any,
) -> None:
    import traceback

    exchange = None
    try:
        iid = InstanceId(instance_id)
        runners = [RunnerId(f"runner-{i}") for i in range(world)]
        shards = {runners[r]: _shard(r, world) for r in range(world)}
        instance = MlxTbcclInstance(
            instance_id=iid,
            shard_assignments=ShardAssignments(
                model_id=ModelId("synthetic/tiny"),
                runner_to_shard=shards,
                node_to_runner={NodeId(node_ids[r]): runners[r] for r in range(world)},
            ),
            hosts_by_node={NodeId(n): "127.0.0.1" for n in node_ids},
        )
        bound = BoundInstance(
            instance=instance,
            bound_runner_id=runners[rank],
            bound_node_id=NodeId(node_ids[rank]),
        )
        exchange = RunnerByteExchange(iid, rank, world, out_s, in_r)
        inner = init_tbccl_pipeline_comm(bound, exchange)
        comm = RecordingComm(inner)
        try:
            res = _run_model(comm, rank, world, dtype_name)
        finally:
            comm.close()
        result_q.put((rank, True, res))
    except BaseException:  # noqa: BLE001
        result_q.put((rank, False, traceback.format_exc()))
    finally:
        if exchange is not None:
            exchange.close()


def _ring_rank(
    rank: int,
    world: int,
    dtype_name: str,
    hostfile: str,
    result_q: Any,
) -> None:
    import traceback

    try:
        os.environ["MLX_HOSTFILE"] = hostfile
        os.environ["MLX_RANK"] = str(rank)
        group = mx.distributed.init(backend="ring", strict=True)
        comm = RecordingComm(MlxPipelineComm(group))
        res = _run_model(comm, rank, world, dtype_name)
        result_q.put((rank, True, res))
    except BaseException:  # noqa: BLE001
        result_q.put((rank, False, traceback.format_exc()))


def _collect(result_q: Any, procs: list[Any], world: int) -> list[dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    deadline = time.time() + 180
    while len(out) + len(errors) < world and time.time() < deadline:
        try:
            rank, ok, value = result_q.get(timeout=1)
        except Exception:
            if all(not p.is_alive() for p in procs):
                codes = [p.exitcode for p in procs]
                raise AssertionError(f"ranks exited without results: {codes}") from None
            continue
        if ok:
            out[rank] = value
        else:
            errors[rank] = value
    for p in procs:
        p.join(5)
        if p.is_alive():
            p.kill()
    assert not errors and len(out) == world, f"errors={errors} got={sorted(out)}"
    return [out[r] for r in range(world)]


def run_tbccl_world(world: int, dtype_name: str) -> list[dict[str, Any]]:
    ctx = mp.get_context()
    instance_id = InstanceId()
    node_ids = [str(NodeId()) for _ in range(world)]
    hubs = [RunnerBytesHub() for _ in range(world)]
    outs: list[MpReceiver[RunnerBytesMessage]] = []
    chans: list[Any] = []
    result_q: Any = ctx.Queue()
    procs: list[Any] = []
    stop = threading.Event()
    for r in range(world):
        out_s, out_r = mp_channel[RunnerBytesMessage]()
        in_s, in_r = mp_channel[RunnerBytesMessage]()
        hubs[r].register(instance_id, in_s)
        outs.append(out_r)
        chans += [out_r, in_s]
        p = ctx.Process(
            daemon=True,
            target=_tbccl_rank,
            args=(
                r,
                world,
                dtype_name,
                node_ids,
                str(instance_id),
                out_s,
                in_r,
                result_q,
            ),
        )
        p.start()
        procs.append(p)

    def pump(rank: int) -> None:
        while not stop.is_set():
            try:
                msg = outs[rank].receive_nowait()
            except Exception:
                time.sleep(0.002)
                continue
            for other, hub in enumerate(hubs):
                if other != rank:
                    hub.deliver(msg)

    threads = [
        threading.Thread(target=pump, args=(r,), daemon=True) for r in range(world)
    ]
    for t in threads:
        t.start()
    try:
        return _collect(result_q, procs, world)
    finally:
        stop.set()
        for c in chans:
            c._state.buffer.cancel_join_thread()
            c.close()


def run_ring_world(world: int, dtype_name: str) -> list[dict[str, Any]]:
    ctx = mp.get_context()
    base_port = 29700 + world * 10
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile = f.name
    result_q: Any = ctx.Queue()
    procs = [
        ctx.Process(
            daemon=True,
            target=_ring_rank,
            args=(r, world, dtype_name, hostfile, result_q),
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    try:
        return _collect(result_q, procs, world)
    finally:
        os.unlink(hostfile)


class _FileExchange:
    """Cross-host variant: the operator (or a sync loop) copies `<purpose>.<rank>` files between hosts."""

    def __init__(self, directory: str, rank: int, world: int, timeout_s: float = 90):
        import pathlib

        self.dir, self.rank, self.world, self.timeout_s = (
            pathlib.Path(directory),
            rank,
            world,
            timeout_s,
        )

    def __call__(self, purpose: str, payload: bytes) -> list[bytes]:
        tmp = self.dir / f".{purpose}.{self.rank}.tmp"
        tmp.write_bytes(payload)
        tmp.replace(self.dir / f"{purpose}.{self.rank}")
        out: list[bytes] = []
        deadline = time.time() + self.timeout_s
        for r in range(self.world):
            f = self.dir / f"{purpose}.{r}"
            while not (f.exists() and f.stat().st_size == len(payload)):
                if time.time() > deadline:
                    raise TimeoutError(f"no {purpose} from rank {r}")
                time.sleep(0.1)
            out.append(f.read_bytes())
        return out


def run_one_host_rank(
    rank: int, world: int, host: str, directory: str, dtype_name: str
) -> dict[str, Any]:
    """One rank of a real multi-host run: bootstraps TbcclPipelineComm directly (blobs via files) and runs the same checks."""
    from exo_tbccl.group import TbcclPipelineComm

    inner = TbcclPipelineComm.create(
        rank,
        world,
        _FileExchange(directory, rank, world),
        bind_host=host,
        advertise_host=host,
        timeout_ms=90000,
    )
    comm = RecordingComm(inner)
    try:
        res = _run_model(comm, rank, world, dtype_name)
        _FileExchange(directory, rank, world)("quiesce", b"x")
    finally:
        comm.close()
    res["device"] = str(mx.default_device())
    res["stats"] = {
        "direct_ops": inner.stats.direct_ops,
        "materialized_copies": inner.stats.materialized_copies,
        "staged_fallback_copies": inner.stats.staged_fallback_copies,
    }
    return res


def main() -> None:
    import sys

    mp.set_start_method(
        "spawn"
    )  # MLX/CUDA must not be forked from a process that imported it; mp_channel follows the default context
    if sys.argv[1] == "host":
        rank, world, host, directory, dtype_name = (
            int(sys.argv[2]),
            int(sys.argv[3]),
            sys.argv[4],
            sys.argv[5],
            sys.argv[6],
        )
        print(
            "RESULT "
            + json.dumps(run_one_host_rank(rank, world, host, directory, dtype_name))
        )
        return
    backend, world, dtype_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    run = run_tbccl_world if backend == "tbccl" else run_ring_world
    print("RESULT " + json.dumps(run(world, dtype_name)))


if __name__ == "__main__":
    main()
