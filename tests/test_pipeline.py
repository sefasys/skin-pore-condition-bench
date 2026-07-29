import json

from pore_assessment.pipeline import _training_complete


def _write_run(tmp_path, epochs_completed, epochs_requested, stopped_early):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"checkpoint marker")
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "configuration": {"epochs_requested": epochs_requested},
                "epochs_completed": epochs_completed,
                "stopped_early": stopped_early,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_training_complete_after_requested_epochs(tmp_path) -> None:
    run_dir = _write_run(tmp_path, 35, 35, False)
    assert _training_complete(run_dir)


def test_training_complete_after_early_stopping(tmp_path) -> None:
    run_dir = _write_run(tmp_path, 24, 35, True)
    assert _training_complete(run_dir)


def test_training_incomplete_when_interrupted(tmp_path) -> None:
    run_dir = _write_run(tmp_path, 12, 35, False)
    assert not _training_complete(run_dir)
