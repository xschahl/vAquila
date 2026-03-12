---
title: CLI Reference
---

## Core commands

- `vaq run <model_id>`
- `vaq ps`
- `vaq stop <model_id> [--purge-cache]`
- `vaq list`
- `vaq rm <model_id>`
- `vaq doctor`
- `vaq infer`
- `vaq ui`

## Run options highlights

- `--device gpu|cpu`
- `--max-num-seqs`
- `--max-model-len`
- `--quantization`
- `--kv-cache-dtype`
- `--trust-remote-code`

Manual overrides:

- `--gpu-utilization <ratio>` in `(0, 1]`
- `--cpu-utilization <ratio>` in `(0, 1]`

When manual overrides are provided, automatic estimation/optimization is bypassed.
