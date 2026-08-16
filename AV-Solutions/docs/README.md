# AV-Solutions Deployment Handoffs

`/home/lixingfeng/UniAD_examine/DL4AGX` is the local root for the three deployment chains:

| Model | Canonical project | Development handoff |
| --- | --- | --- |
| UniAD-tiny/base | `AV-Solutions/uniad-trt` | `docs/4090-UniAD-tiny.md` |
| VAD-Tiny | `AV-Solutions/vad-trt` | `docs/4090-VAD.md` |
| UniV2X | `AV-Solutions/univ2x-trt` | `docs/4090-UniV2X.md` |

The old `UniV2X/deploy_int8` path is retained as a compatibility symlink to the canonical UniV2X deployment directory. The large `artifacts` directory remains outside the source handoff through a local symlink. Original datasets and checkpoints under `/data` are never copied, modified, or deleted.

The three handoff documents are separate because their input contracts, temporal state protocols, calibration feeds, and task metrics are different. They share the same root-level layout and source-control exclusion policy, but their results must not be merged into one accuracy table.
