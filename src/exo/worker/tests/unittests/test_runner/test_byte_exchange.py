import threading
import time

import pytest

from exo.shared.types.runner_bytes import RunnerBytesMessage
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import MpReceiver, MpSender, WouldBlock, mp_channel
from exo.worker.runner.byte_exchange import RunnerByteExchange, RunnerByteExchangeError
from exo.worker.runner_bytes_hub import RunnerBytesHub


class FakeCluster:
    """N runners, one Worker hub each, joined by a lossy 'zenoh' that drops messages to ranks that are not started yet."""

    def __init__(self, world: int, instance_id: InstanceId):
        self.world = world
        self.instance_id = instance_id
        self.hubs = [RunnerBytesHub() for _ in range(world)]
        self.chans: list[
            MpSender[RunnerBytesMessage] | MpReceiver[RunnerBytesMessage]
        ] = []
        self.exchanges: dict[int, RunnerByteExchange] = {}
        self.started: set[int] = set()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.dropped = 0

    def start_rank(self, rank: int) -> RunnerByteExchange:
        out_s, out_r = mp_channel[RunnerBytesMessage]()
        in_s, in_r = mp_channel[RunnerBytesMessage]()
        self.chans += [out_s, out_r, in_s, in_r]
        self.hubs[rank].register(self.instance_id, in_s)
        ex = RunnerByteExchange(self.instance_id, rank, self.world, out_s, in_r)
        self.exchanges[rank] = ex
        self.started.add(rank)

        def pump() -> None:
            while not self._stop.is_set():
                try:
                    msg = out_r.receive_nowait()
                except Exception:
                    time.sleep(0.002)
                    continue
                for r, hub in enumerate(self.hubs):
                    if r == rank:
                        continue
                    if r in self.started:
                        hub.deliver(msg)
                    else:
                        self.dropped += 1

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        self._threads.append(t)
        return ex

    def stop(self) -> None:
        self._stop.set()
        for ex in self.exchanges.values():
            ex.close()
        for t in self._threads:
            t.join(2)
        for c in self.chans:
            c._state.buffer.cancel_join_thread()  # pyright: ignore[reportPrivateUsage]
            c.close()


def _gather_all(
    cluster: FakeCluster,
    namespace: str,
    payloads: list[bytes],
    *,
    delay: dict[int, float] | None = None,
    timeout_s: float = 10,
):
    results: dict[int, list[bytes] | BaseException] = {}

    def run(rank: int) -> None:
        time.sleep((delay or {}).get(rank, 0))
        ex = cluster.start_rank(rank)
        try:
            results[rank] = ex.all_gather(
                namespace, payloads[rank], timeout_s=timeout_s
            )
        except BaseException as e:  # noqa: BLE001
            results[rank] = e

    ts = [threading.Thread(target=run, args=(r,)) for r in range(cluster.world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


@pytest.mark.parametrize("world", [2, 3, 4])
def test_all_gather_returns_every_payload_in_rank_order(world: int):
    cluster = FakeCluster(world, InstanceId())
    try:
        payloads = [bytes([r]) * (r + 1) for r in range(world)]
        res = _gather_all(cluster, "tbccl/endpoint", payloads)
        for r in range(world):
            assert res[r] == payloads
    finally:
        cluster.stop()


def test_a_rank_that_starts_late_still_completes_because_peers_republish():
    cluster = FakeCluster(3, InstanceId())
    try:
        payloads = [b"a", b"bb", b"ccc"]
        res = _gather_all(cluster, "ns", payloads, delay={2: 1.2})
        assert cluster.dropped > 0  # the early messages really were lost
        assert all(res[r] == payloads for r in range(3))
    finally:
        cluster.stop()


def test_completed_ranks_keep_answering_a_slow_peer():
    cluster = FakeCluster(2, InstanceId())
    try:
        e0 = cluster.start_rank(0)
        # rank 1 is slow: rank 0 can only finish after rank 1 publishes, then rank 1 must still get rank 0's payload
        t = threading.Thread(target=lambda: e0.all_gather("ns", b"zero", timeout_s=10))
        t.start()
        time.sleep(0.8)
        e1 = cluster.start_rank(1)
        assert e1.all_gather("ns", b"one", timeout_s=10) == [b"zero", b"one"]
        t.join()
    finally:
        cluster.stop()


def test_messages_from_another_instance_never_match():
    inst, other = InstanceId(), InstanceId()
    cluster = FakeCluster(2, inst)
    try:
        e0 = cluster.start_rank(0)
        stale = RunnerBytesMessage(
            instance_id=other,
            namespace="ns",
            rank=1,
            world_size=2,
            payload_b64="c3RhbGU=",
        )
        cluster.hubs[0].register(other, cluster.hubs[0]._inboxes[inst])  # pyright: ignore[reportPrivateUsage]
        cluster.hubs[0].deliver(stale)
        with pytest.raises(TimeoutError):
            e0.all_gather("ns", b"x", timeout_s=1.0)
    finally:
        cluster.stop()


def test_timeout_is_bounded_and_names_the_missing_ranks():
    cluster = FakeCluster(3, InstanceId())
    try:
        e0 = cluster.start_rank(0)
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match=r"ranks \[1, 2\]"):
            e0.all_gather("ns", b"x", timeout_s=0.5)
        assert time.monotonic() - t0 < 3
    finally:
        cluster.stop()


def test_cancellation_unblocks_the_exchange():
    cluster = FakeCluster(2, InstanceId())
    try:
        e0 = cluster.start_rank(0)
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        t0 = time.monotonic()
        with pytest.raises(RunnerByteExchangeError, match="cancelled"):
            e0.all_gather("ns", b"x", timeout_s=30, should_abort=flag.is_set)
        assert time.monotonic() - t0 < 3
    finally:
        cluster.stop()


def test_closed_inbox_unblocks_the_exchange():
    cluster = FakeCluster(2, InstanceId())
    try:
        e0 = cluster.start_rank(0)
        inbox_sender = cluster.hubs[0]._inboxes[cluster.instance_id]  # pyright: ignore[reportPrivateUsage]
        threading.Timer(0.3, inbox_sender.close).start()
        with pytest.raises(RunnerByteExchangeError, match="closed"):
            e0.all_gather("ns", b"x", timeout_s=30)
    finally:
        cluster.stop()


def test_hub_buffers_messages_for_a_runner_that_does_not_exist_yet_and_is_bounded():
    hub = RunnerBytesHub()
    inst = InstanceId()
    for _ in range(1000):
        hub.deliver(
            RunnerBytesMessage(
                instance_id=inst,
                namespace="n",
                rank=1,
                world_size=2,
                payload_b64="AA==",
            )
        )
    s, r = mp_channel[RunnerBytesMessage]()
    hub.register(inst, s)
    got: list[RunnerBytesMessage] = []
    deadline = time.monotonic() + 3
    while len(got) < 256 and time.monotonic() < deadline:
        got += r.collect()
        time.sleep(0.01)
    assert len(got) == 256  # bounded: 1000 early messages keep only the first 256
    s._state.buffer.cancel_join_thread()  # pyright: ignore[reportPrivateUsage]
    s.close()
    r.close()
    hub.unregister(inst)
    hub.deliver(
        RunnerBytesMessage(
            instance_id=InstanceId(),
            namespace="n",
            rank=0,
            world_size=2,
            payload_b64="AA==",
        )
    )


def test_inbox_keeps_being_drained_after_the_exchange_completes():
    """Phase 68 (real exo + TBCCL run): after the bootstrap exchange a peer keeps republishing its (several KB) endpoint blob for its own linger period. If the
    runner stops reading its inbox the OS pipe fills, the mp.Queue feeder thread in the Worker blocks, and the exo process then hangs at interpreter exit
    (multiprocessing's Queue._finalize_join). The exchange must keep consuming (and discarding) late messages until it is closed."""
    inst = InstanceId()
    out_s, out_r = mp_channel[RunnerBytesMessage]()
    in_s, in_r = mp_channel[RunnerBytesMessage]()
    ex = RunnerByteExchange(inst, 0, 2, out_s, in_r)
    try:
        late = RunnerBytesMessage(instance_id=inst, namespace="ns", rank=1, world_size=2, payload_b64="A" * 6000)
        in_s.send(late)
        got = ex.all_gather("ns", b"zero", timeout_s=5)
        assert got[0] == b"zero" and len(got[1]) == 4500
        for _ in range(60):  # ~360 KB, far beyond a 64 KB pipe
            in_s.send(late)
        time.sleep(3.5)  # the linger loop wakes every RESEND_INTERVAL_S * 2 = 1 s
        with pytest.raises(WouldBlock):
            in_r.receive_nowait()  # nothing is left in the queue: the exchange consumed the late messages itself
    finally:
        ex.close()
        for ch in (out_s, out_r, in_s, in_r):
            ch.close()
