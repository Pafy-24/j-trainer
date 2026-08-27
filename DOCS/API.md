# API și CLI Trainer

Serverul FastAPI este disponibil implicit pe portul primit din lease. Toate
mutațiile acceptă `Idempotency-Key`; PATCH/DELETE folosesc `If-Match`.

Resursele publice sunt dataseturi, campanii, joburi, operații și rezultate.
Endpointurile principale sunt cele din `/v1/datasets`, `/v1/campaigns`,
`/v1/jobs`, `/v1/results`, `/v1/pool`, `/v1/workers` și `/v1/events/stream`.
Contractul complet este generat în `schemas/openapi.json`.

CLI:

```text
j-train dataset register|validate|shard-plan|inspect
j-train campaign apply|plan|start|pause|resume|cancel|status|inspect
j-train pool status|workers|metrics|watch
j-train job list|inspect|retry
j-train result list|inspect|promote|export
j-train migrate-mi-config
j-train migrate-mi-checkpoint
```

CLI poate citi `JESTER_TRAINER_URL` și `JESTER_TRAINER_TOKEN_FILE`; serverul
însuși nu pornește fără lease/topologie RM.
