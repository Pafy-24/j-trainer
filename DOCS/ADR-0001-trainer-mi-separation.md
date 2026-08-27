# ADR-0001 — Separarea Trainer–MI

- Data: 27.08.2026
- Stare: acceptat
- Componentă: Jester 5/5

## Decizie

Trainer deține dataseturile, campaniile, DAG/sweep, optimizerul, schedulerul,
agregarea și checkpointurile. `mi.worker` deține numai model load, cache-ul
local și gradientul copilului. RM deține procesele, resursele, pool-ul, CAS și
canalele. LSR nu intră în fluxul de training.

Rundele parameter-server sunt sincrone. Contribuțiile sunt ponderate cu numărul
real de tokeni supravegheați. Tensor names/shapes, versiunea, hash-ul și valorile
finite sunt validate înainte de update. Baza și părinții nu circulă per rundă.

## Consecințe

- `mi.train` nu mai este adapter activ.
- `/training/jobs` răspunde temporar `trainer_required`.
- Configurile se migrează cu `j-train migrate-mi-config`.
- Ponderile legacy devin `checkpoint.initialWeights` prin
  `j-train migrate-mi-checkpoint`; optimizerul/schedulerul repornesc explicit.
- Trainer nu are mod standalone și cere topologie/lease RM.
