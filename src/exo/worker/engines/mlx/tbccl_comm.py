"""Builds the TBCCL pipeline communicator for an MlxTbccl instance (optional exo-tbccl package).

exo owns membership, rank and addresses: the rank and world size come straight from the PipelineShardMetadata, the address from the instance
(chosen at placement from exo's topology), and the bootstrap blobs travel through exo's generic runner byte exchange as opaque bytes.
"""

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

from exo.shared.types.worker.instances import BoundInstance, MlxTbcclInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.pipeline_comm import PipelineComm
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.byte_exchange import RunnerByteExchange

BOOTSTRAP_TIMEOUT_S = 120.0


class _TbcclPipelineCommFactory(Protocol):
    @classmethod
    def create(
        cls,
        rank: int,
        world_size: int,
        exchange: "_Exchange",
        *,
        bind_host: str | None = None,
        advertise_host: str | None = None,
        timeout_ms: int = 0,
    ) -> PipelineComm: ...


class _Exchange(Protocol):
    def __call__(self, purpose: str, payload: bytes) -> Sequence[bytes]: ...


class _GroupModule(Protocol):
    TbcclPipelineComm: type[_TbcclPipelineCommFactory]


def _import(name: str) -> object:
    return import_module(name)


def init_tbccl_pipeline_comm(
    bound_instance: BoundInstance, byte_exchange: RunnerByteExchange | None
) -> PipelineComm:
    instance = bound_instance.instance
    assert isinstance(instance, MlxTbcclInstance)
    shard = bound_instance.bound_shard
    if not isinstance(shard, PipelineShardMetadata):
        raise RuntimeError("MlxTbccl supports pipeline shards only")
    if byte_exchange is None:
        raise RuntimeError(
            "MlxTbccl needs the runner byte exchange (not provided by this runner)"
        )
    try:
        module = cast(_GroupModule, _import("exo_tbccl.group"))
    except ImportError as e:
        raise RuntimeError(
            f"MlxTbccl selected but the optional exo-tbccl package cannot be imported: {e}"
        ) from e
    host = instance.hosts_by_node[bound_instance.bound_node_id]
    rank, world = shard.device_rank, shard.world_size
    iid = instance.instance_id
    logger.info(f"rank {rank}/{world} TBCCL bootstrap on {host}")

    def exchange(purpose: str, payload: bytes) -> Sequence[bytes]:
        return byte_exchange.all_gather(
            f"tbccl/{purpose}", payload, timeout_s=BOOTSTRAP_TIMEOUT_S
        )

    comm = module.TbcclPipelineComm.create(
        rank,
        world,
        exchange,
        bind_host=host,
        advertise_host=host,
        timeout_ms=int(BOOTSTRAP_TIMEOUT_S * 1000),
    )
    logger.info(f"rank {rank}/{world} TBCCL communicator ready (instance {iid})")
    return comm
