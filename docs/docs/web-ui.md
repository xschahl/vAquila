---
title: Web UI
---

## Start Web UI

```bash
docker compose run --rm -p 8787:8787 vaq ui --host 0.0.0.0 --port 8787
```

Open: `http://localhost:8787`

## Preview

![vAquila Control Center preview](/img/control-center-ui.png)

## Capabilities

- Launch models in GPU or CPU mode
- Follow run tasks and logs
- Stop containers and clear cache
- Inspect GPU usage and host CPU/RAM usage
- Run quick inference checks

## Manual utilization mode

In the Run form, you can set:

- `GPU utilization ratio` (GPU mode)
- `CPU utilization ratio` (CPU mode)

When one is set, estimation and auto-optimization are disabled for that launch.
