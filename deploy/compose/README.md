# Single-machine development deployment

Copy `.env.example` to `.env`, replace every development credential, and run:

```bash
docker compose --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml up --build
```

The control services run locally and the WebUI listens on port 3000. A real
vLLM worker must be started from the separately pinned worker image and exposed
through `FREECHAT_DEFAULT_WORKER_ENDPOINT`. This deployment does not claim HA,
SPIFFE identity, or a validated GPU data plane.
