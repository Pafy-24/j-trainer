from __future__ import annotations

from pathlib import Path

from jester_trainer.config import Settings
from jester_trainer.service import TrainerService


class _RMWithMetrics:
    available = True

    @staticmethod
    def metrics() -> dict:
        common = {"hostId": "host-a", "leaseId": "lease-a", "sampledAt": "2026-08-27T10:00:00Z"}
        return {
            "samples": [
                {**common, "instanceId": "worker-a", "pid": 10, "metric": "process.cpu.percent", "value": 12},
                {**common, "instanceId": "worker-a", "pid": 10, "metric": "process.memory.rss", "value": 100},
                {**common, "instanceId": "worker-a", "pid": 10, "resourceId": "gpu-a", "metric": "accelerator.memory.allocated", "value": 100},
                {**common, "instanceId": "worker-a", "pid": 10, "resourceId": "gpu-a", "metric": "accelerator.memory.reserved", "value": 200},
                {**common, "instanceId": "worker-a", "pid": 10, "resourceId": "gpu-a", "metric": "process.accelerator.memory.used", "value": 150},
                {**common, "instanceId": "test-trainer", "pid": 20, "metric": "process.cpu.percent", "value": 5},
                {**common, "instanceId": "test-trainer", "pid": 20, "metric": "process.memory.rss", "value": 50},
                {**common, "instanceId": "test-trainer", "pid": 20, "resourceId": "gpu-b", "metric": "accelerator.memory.reserved", "value": 25},
            ],
            "physicalCapacity": [
                {"hostId": "host-a", "resourceId": "gpu-a", "capacity": {"memoryBytes": 1000}}
            ],
            "channels": [],
        }


def test_pool_metrics_sum_processes_without_double_counting_accelerator_sources(
    tmp_path: Path,
) -> None:
    service = TrainerService(Settings.for_test(tmp_path / "trainer"))
    service.rm = _RMWithMetrics()  # type: ignore[assignment]
    try:
        result = service.pool_metrics()
        assert result["topologyMeasured"] == {
            "cpuPercent": 17.0,
            "ramBytes": 150,
            "acceleratorMemoryBytes": 225,
        }
        assert result["trainerIncludedByRM"] is True
        assert result["trainer"] == {"status": "external", "authority": "RMA"}
        assert result["physicalCapacity"][0]["hostId"] == "host-a"
    finally:
        service.close()


def test_unavailable_rm_does_not_turn_missing_measurements_into_zero(tmp_path: Path) -> None:
    service = TrainerService(Settings.for_test(tmp_path / "trainer"))
    try:
        result = service.pool_metrics()
        assert result["topologyMeasured"] is None
        assert result["source"] == "unavailable"
    finally:
        service.close()
