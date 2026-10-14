# MlxTbccl: pipeline parallelism over TBCCL (optional, experimental)

`MlxTbccl` is an optional instance type that moves pipeline-stage activations with
[TBCCL](https://github.com/tensorsofthewall/tbccl) through the separate `exo-tbccl` package instead of `mx.distributed`. It lets a Mac (Metal) and a Linux machine
(CUDA) run one pipeline over a Thunderbolt or Ethernet link. exo still owns discovery, placement and shard assignment; TBCCL only moves bytes.

## Selecting it

```sh
curl -X POST localhost:52415/place_instance -H 'content-type: application/json' \
  -d '{"model_id": "mlx-community/Qwen3-0.6B-8bit", "sharding": "Pipeline", "instance_meta": "MlxTbccl", "min_nodes": 2}'
```

Placement uses exactly the layer allocation of the other runtimes. Each node advertises the address every other node uses to reach it (the
thunderbolt interface is preferred, as for the ring backend).

## Requirements

- The `exo-tbccl` package in exo's environment (`exo_tbccl.is_available()`); its wheel already contains TBCCL (C ABI 1, wire protocol 4), so nothing else is installed. See [TBCCL-FORK.md](../TBCCL-FORK.md). exo runs normally
  without it and `MlxTbccl` placement then fails with a message saying what is missing. The package is published as GitHub release wheels (not on PyPI) and is
  not a mandatory exo dependency; `tbccl-integration-v2` offers it as the optional `tbccl` extra (`uv sync --locked --extra mlx-cuda13 --extra tbccl`, or `--extra mlx --extra tbccl` on macOS; Linux x86-64 and macOS arm64 only).
- Backends: MlxMetal, MlxCuda, MlxCpu (a Mac and a Linux/CUDA node can be mixed).

## Upstream base

The integration is maintained as a short series of commits over upstream exo (`75d8ad2b`, 2026-10-12, for `tbccl-integration-v1`; `bc412e22`, 2026-10-13, for `tbccl-integration-v2`, which also adds the optional `tbccl` extra): the `PipelineComm` seam, the `MlxTbccl` instance type and its placement rules, the generic runner byte exchange, `TbcclPipelineComm`, and the synthetic pipeline driver. Without the optional package exo behaves as upstream.

## Limitations

- Pipeline sharding and text generation only: no tensor parallelism (rejected at placement), no image / CFG models.
- One advertised address per node: topologies that need different addresses for different peers are rejected.
- Experimental: nothing here changes the default runtimes.

## For developers

- `exo.worker.engines.mlx.pipeline_comm`: the `PipelineComm` protocol; `MlxPipelineComm` wraps `mx.distributed.Group` unchanged.
- `exo.worker.runner.byte_exchange` / `runner_bytes_hub`: a generic all-gather of opaque bytes between the runners of an instance (zenoh topic
  `runner_bytes`). Any backend that needs bootstrap data can use it.
- `exo.worker.engines.mlx.tbccl_comm`: builds the TBCCL communicator from the instance.
- Tests: `test_pipeline_comm.py`, `test_byte_exchange.py`, `test_placement.py` (default suite); `test_tbccl_pipeline.py` (slow, needs `exo-tbccl`).
