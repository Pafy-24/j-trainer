# Jester Trainer (componenta 5/5)

Trainer este coordonatorul RM-managed pentru antrenarea fațetelor Mixed
Inference. El deține manifestele dataseturilor, campaniile, DAG-urile și
sweep-urile, optimizerul, schedulerul, rundele sincrone și checkpointurile.
Workerii `mi.worker` calculează numai gradienții fațetei copil; RM deține
procesele, lease-urile, resursele și canalele.

Trainer nu are mod standalone și nu apelează LSR. Serverul pornește numai
dintr-un lease `trainer.serve` aflat într-o topologie RM de tip `training`:

```text
jester-rm topology plan training-small
jester-rm topology start training-small
j-train pool status
j-train campaign apply examples/campaign-two-workers.json
j-train campaign plan demo-coder
j-train campaign start demo-coder
```

Pentru comenzile CLI lansate în afara procesului Trainer se configurează
endpointul și tokenul emise în volumul instanței:

```text
JESTER_TRAINER_URL=http://127.0.0.1:8140
JESTER_TRAINER_TOKEN_FILE=<instance-root>/run/trainer/trainer.token
```

Contractele și limitele de platformă sunt documentate în `DOCS/`, iar
schemele generate se află în `schemas/`.

## Resurse instalate pentru daemon

Distribuția include `jester_trainer/resources/component-manifest.json` și
resursele declarative fixate prin hash pentru capabilitatea gestionată
`trainer.campaign`. Configurația reutilizabilă a daemonului este exact `{}`:
hostul, portul, topologia de tip `training`, rolul `trainer`, lease-ul, căile și
tokenul provin din RM. Campaniile, dataseturile, seed-urile, optimizerul,
workerii și checkpointurile rămân payload-uri explicite ale operațiilor lor;
manifestul daemonului nu le configurează și nu le autorizează.
