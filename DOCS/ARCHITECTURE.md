# Arhitectura Trainer

Trainer este componenta 5/5 și rulează numai în nodul `trainer.serve` al unei
topologii RM `training`.

```text
Campaign DAG -> scheduler -> cohortă -> sesiuni MI -> gradients
CPU optimizer/scheduler <- agregare token-weighted
                    -> checkpoint atomic -> artifact MI
```

`independent` atribuie câte un worker joburilor independente și permite pool-uri
eterogene. `parameter-server` cere cohortă omogenă și runde sincrone. Pierderea
unui worker anulează runda și retry-ul reia ultimul checkpoint confirmat.

Dataseturile sunt sharded după digest, seed, split și `semanticCaseId`.
Sursele remote, modelul și registrele de părinți sunt bindinguri CAS aprobate de
RM. MI rezolvă numai câmpurile permise și refuză digesturile absente.
