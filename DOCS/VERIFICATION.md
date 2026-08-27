# Verificare Trainer

## Gate local închis

- API/CLI și SQLite WAL;
- DatasetManifest, shard deterministic și interdicția test split;
- parameter server cu unul și doi workeri CPU;
- paritate numerică doi workeri versus batch combinat;
- gradient copil-only, idempotency, stale version și NaN/Inf checks;
- idempotency REST stabilă peste timestampurile generate de server;
- bindinguri CAS reaplicate după recovery și publicare dinamică prin RM;
- totaluri Trainer + workers fără dublarea surselor VRAM;
- checkpoint atomic și artifact MI;
- migrare config și ponderi legacy.

## Gate-uri necalificate încă pe hardware real

- mai mulți workeri CUDA pe RTX 5070;
- combinațiile Trainer/worker prin SSH și reconnect de transfer mare;
- ROCm/HIP, DirectML autograd și TPU/XLA;
- fault injection remote în mijlocul rundei.

`hardwareVerified` rămâne fals până la gate-ul corespunzător.
