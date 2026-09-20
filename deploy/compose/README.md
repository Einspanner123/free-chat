# Single-machine development deployment

Copy `.env.example` to `.env`, replace every development credential, and run:

```bash
docker compose --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml up --build
```

The control services run locally and the WebUI listens on port 3000. A real
vLLM worker must be started from the separately pinned worker image and exposed
through an address reachable by the Gateway. Compose enables the gRPC Scheduler,
so `FREECHAT_DEFAULT_WORKER_ENDPOINT` alone does not register a Worker. A managed
Worker registration and continuous heartbeat process is still required; the
deployment reconciler is not yet supplied by this Compose file. Do not interpret
the control-service startup command as a complete inference bootstrap.

Set `FREECHAT_ORIGIN_NODE_ID` to the administrator-defined routing origin, matching
a Worker `node_id`. It is not taken from browser/client headers. When absent,
requests forbidding remote execution are rejected; permissive requests cannot use
a locality-dependent cost estimate and fall back explicitly.

The Scheduler requires model KV block size, worst-rank KV bytes/token, and a
generation-bound heartbeat reporting the minimum usable KV admission budget across
ranks. Missing values reject admission. Existing generic telemetry sampling does
not derive these values from CUDA free memory. See `docs/review-hardening.md` for
the pending Worker integration. This deployment does not claim HA, SPIFFE identity,
or a validated GPU data plane.
