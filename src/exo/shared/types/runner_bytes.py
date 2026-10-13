from exo.shared.types.worker.instances import InstanceId
from exo.utils.pydantic_ext import FrozenModel


class RunnerBytesMessage(FrozenModel):
    """One rank's opaque payload for a generic all-gather between the runners of an instance.

    The contents are never interpreted by exo. ``namespace`` separates independent exchanges (e.g. a backend's bootstrap rounds); the
    instance id keeps a new instance from ever matching a previous instance's messages.
    """

    instance_id: InstanceId
    namespace: str
    rank: int
    world_size: int
    payload_b64: str
