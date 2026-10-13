"""Worker-side routing of runner byte-exchange messages to the local runner of the matching instance.

Messages that arrive before the local runner exists are held (bounded) and handed over when it registers, so a fast peer cannot lose its
payload to a slow-starting node. Unknown instances are not kept forever: the buffer is capped per instance and in total.
"""

from exo.shared.types.runner_bytes import RunnerBytesMessage
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import ClosedResourceError, MpSender, WouldBlock

MAX_BUFFERED_PER_INSTANCE = 256
MAX_BUFFERED_INSTANCES = 64


class RunnerBytesHub:
    def __init__(self) -> None:
        self._inboxes: dict[InstanceId, MpSender[RunnerBytesMessage]] = {}
        self._early: dict[InstanceId, list[RunnerBytesMessage]] = {}

    def register(
        self, instance_id: InstanceId, inbox: MpSender[RunnerBytesMessage]
    ) -> None:
        self._inboxes[instance_id] = inbox
        for msg in self._early.pop(instance_id, []):
            self._send(instance_id, msg)

    def unregister(self, instance_id: InstanceId) -> None:
        self._inboxes.pop(instance_id, None)
        self._early.pop(instance_id, None)

    def deliver(self, msg: RunnerBytesMessage) -> None:
        if msg.instance_id in self._inboxes:
            self._send(msg.instance_id, msg)
            return
        if (
            msg.instance_id not in self._early
            and len(self._early) >= MAX_BUFFERED_INSTANCES
        ):
            return
        buf = self._early.setdefault(msg.instance_id, [])
        if len(buf) < MAX_BUFFERED_PER_INSTANCE:
            buf.append(msg)

    def _send(self, instance_id: InstanceId, msg: RunnerBytesMessage) -> None:
        inbox = self._inboxes.get(instance_id)
        if inbox is None:
            return
        try:
            inbox.send_nowait(msg)
        except ClosedResourceError:
            self._inboxes.pop(instance_id, None)
        except WouldBlock:
            pass  # a full inbox only drops a message: senders republish
