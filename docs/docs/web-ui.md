---
title: Web UI
---

## Start Web UI

You can start the Web UI easily using the [dedicated Docker examples on GitHub](https://github.com/xschahl/vaquila/tree/main/docs/examples/webui) or run it locally with the official image:

```bash
docker compose -f docs/examples/webui/docker-compose.yml up -d
```

Open your browser at: `http://localhost:8787`

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
