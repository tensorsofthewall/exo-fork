"""Generic all-gather of opaque bytes between the runners of one instance (runner side).

A runner only talks to its own Worker, so the exchange goes runner -> Worker (outbox) -> zenoh topic -> peer Workers -> peer runners (inbox).
The pub/sub layer is best effort (a peer may not be subscribed yet), so every rank republishes its payload until the gather completes and
keeps answering late peers for a bounded linger period. Everything is bounded by a timeout and ends when the inbox is closed.
"""

import base64
import contextlib
import threading
import time
from collections.abc import Callable

from exo.shared.types.runner_bytes import RunnerBytesMessage
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import (
    ClosedResourceError,
    EndOfStream,
    MpReceiver,
    MpSender,
    WouldBlock,
)

RESEND_INTERVAL_S = 0.5
LINGER_S = 60.0
POLL_S = 0.005


class RunnerByteExchangeError(RuntimeError):
    pass


class RunnerByteExchange:
    def __init__(
        self,
        instance_id: InstanceId,
        rank: int,
        world_size: int,
        outbox: MpSender[RunnerBytesMessage],
        inbox: MpReceiver[RunnerBytesMessage],
    ):
        self.instance_id = instance_id
        self.rank = rank
        self.world_size = world_size
        self._outbox = outbox
        self._inbox = inbox
        self._lock = threading.Lock()
        self._received: dict[str, dict[int, bytes]] = {}
        self._completed: dict[
            str, bytes
        ] = {}  # my payload per finished namespace, for late peers
        self._last_complete: float | None = None
        self._stop = threading.Event()
        self._linger = threading.Thread(target=self._linger_loop, daemon=True)
        self._linger.start()

    def _publish(self, namespace: str, payload: bytes) -> None:
        msg = RunnerBytesMessage(
            instance_id=self.instance_id,
            namespace=namespace,
            rank=self.rank,
            world_size=self.world_size,
            payload_b64=base64.b64encode(payload).decode("ascii"),
        )
        self._outbox.send(msg)

    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self._inbox.receive_nowait()
            except WouldBlock:
                return
            except (ClosedResourceError, EndOfStream) as e:
                raise RunnerByteExchangeError("runner byte exchange closed") from e
            if msg.instance_id != self.instance_id or msg.world_size != self.world_size:
                continue  # another instance (or a stale one): never matches this communicator
            if not 0 <= msg.rank < self.world_size or msg.rank == self.rank:
                continue
            with self._lock:
                if msg.namespace in self._completed:
                    continue
                self._received.setdefault(msg.namespace, {})[msg.rank] = (
                    base64.b64decode(msg.payload_b64)
                )

    def all_gather(
        self,
        namespace: str,
        payload: bytes,
        *,
        timeout_s: float,
        should_abort: Callable[[], bool] = lambda: False,
    ) -> list[bytes]:
        """Block until every rank of the instance has published ``payload`` under ``namespace``; returns payloads in rank order."""
        deadline = time.monotonic() + timeout_s
        next_send = 0.0
        while True:
            now = time.monotonic()
            if now >= next_send:
                self._publish(namespace, payload)
                next_send = now + RESEND_INTERVAL_S
            self._drain_inbox()
            with self._lock:
                got = self._received.get(namespace, {})
                if len(got) == self.world_size - 1:
                    out = [
                        payload if r == self.rank else got[r]
                        for r in range(self.world_size)
                    ]
                    self._received.pop(namespace, None)
                    self._completed[namespace] = payload
                    self._last_complete = time.monotonic()
                    return out
                missing = sorted(set(range(self.world_size)) - set(got) - {self.rank})
            if should_abort():
                raise RunnerByteExchangeError(
                    f"exchange {namespace!r} cancelled (waiting for ranks {missing})"
                )
            if now >= deadline:
                raise TimeoutError(
                    f"exchange {namespace!r} timed out after {timeout_s}s waiting for ranks {missing}"
                )
            time.sleep(POLL_S)

    def _linger_loop(self) -> None:
        # Answer late peers for a bounded time, but keep consuming the inbox until close(): a peer republishes its (several KB) payload for its own linger
        # period, and an unread inbox fills the OS pipe, blocks the Worker's mp.Queue feeder thread and hangs the exo process at interpreter exit.
        while not self._stop.wait(RESEND_INTERVAL_S * 2):
            try:
                self._drain_inbox()
            except RunnerByteExchangeError:
                return
            with self._lock:
                done = list(self._completed.items())
                last = self._last_complete
            if last is not None and time.monotonic() - last > LINGER_S:
                continue
            for namespace, payload in done:
                with contextlib.suppress(ClosedResourceError):
                    self._publish(namespace, payload)

    def close(self) -> None:
        self._stop.set()
