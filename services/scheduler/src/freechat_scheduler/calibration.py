from datetime import UTC, datetime

from freechat_contracts import CostCalibration, RequestProfile

from freechat_scheduler.registry import WorkerSnapshot


def matching_calibration(
    request: RequestProfile,
    worker: WorkerSnapshot,
) -> tuple[CostCalibration | None, str]:
    model = next(item for item in worker.capabilities.models if item.model_id == request.model_id)
    now = datetime.now(UTC)
    for calibration in sorted(
        worker.telemetry.calibrations,
        key=lambda item: item.observed_at,
        reverse=True,
    ):
        if (
            calibration.worker_id != worker.capabilities.worker_id
            or calibration.worker_generation != worker.capabilities.generation
            or calibration.engine_instance_id != worker.telemetry.engine_instance_id
            or calibration.model != model
        ):
            continue
        if not calibration.observed_at <= now < calibration.expires_at:
            continue
        if (now - calibration.observed_at).total_seconds() > 3600:
            continue
        if not calibration.input_tokens_min <= request.input_tokens <= calibration.input_tokens_max:
            continue
        if (
            not calibration.output_tokens_min
            <= request.output_tokens
            <= calibration.output_tokens_max
        ):
            continue
        if worker.telemetry.active_requests + 1 > calibration.max_concurrent_requests:
            continue
        # Queueing was not measured by the initial isolated-request profiler.
        if worker.telemetry.queue_depth > 0 and calibration.max_concurrent_requests == 1:
            continue
        return calibration, "matched"
    return None, "no_fresh_calibration_for_request_scope"
