from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from model_passport.core import capture
from model_passport.core.builder import BuildError, build_passport
from model_passport.core.config import CONFIG_FILENAME, StageConfig, TrackingConfig, load_config
from model_passport.core.tracking import (
    MlflowTracker,
    TrackingUnavailableError,
    dvc_add,
    resolve_tracking_uri,
)
from model_passport.runtime import params

STAGE_ONE = """\
from pathlib import Path
from model_passport.runtime import params
p = params({"factor": 1})
Path("mid.txt").write_text(str(int(Path("in.txt").read_text()) * p["factor"]))
"""

STAGE_TWO = """\
import json
from pathlib import Path
value = int(Path("mid.txt").read_text())
Path("model.bin").write_bytes(bytes([value % 256]))
Path("metrics.json").write_text(json.dumps({"test": {"value": value}}))
"""


@pytest.fixture
def pipeline_project(project: Path) -> Path:
    (project / "in.txt").write_text("21")
    (project / "one.py").write_text(STAGE_ONE)
    (project / "two.py").write_text(STAGE_TWO)
    config = yaml.safe_load((project / CONFIG_FILENAME).read_text())
    config["stages"] = [
        {
            "name": "one",
            "cmd": "python one.py",
            "params": {"factor": 2},
            "deps": ["in.txt"],
            "outs": ["mid.txt"],
        },
        {
            "name": "two",
            "cmd": ["python", "two.py"],
            "deps": ["mid.txt"],
            "outs": ["model.bin", "metrics.json"],
            "metrics": "metrics.json",
        },
    ]
    config["build"]["model"]["path"] = "model.bin"
    (project / CONFIG_FILENAME).write_text(yaml.safe_dump(config))
    return project


def _run(root: Path, **kwargs: object) -> capture.RunRecord:
    record = capture.run_pipeline(root, load_config(root / CONFIG_FILENAME), **kwargs)
    capture.save_run_record(root, record)
    return record


def test_parse_overrides_types_values() -> None:
    assert capture.parse_overrides(["train.depth=3", "train.fast=true", "prep.name=x=y"]) == {
        "train": {"depth": 3, "fast": True},
        "prep": {"name": "x=y"},
    }


@pytest.mark.parametrize("bad", ["nodot=1", "stage.key", ".key=1", "stage.=1"])
def test_parse_overrides_rejects_malformed(bad: str) -> None:
    with pytest.raises(capture.CaptureError):
        capture.parse_overrides([bad])


def test_stage_script_inference() -> None:
    stage = StageConfig(name="s", cmd="python -u demo/train.py --fast")
    assert capture.stage_script(stage) == Path("demo/train.py")
    with pytest.raises(capture.CaptureError):
        capture.stage_script(StageConfig(name="s", cmd="make all"))
    assert capture.stage_script(StageConfig(name="s", cmd="make", script=Path("Makefile"))) == (
        Path("Makefile")
    )


def test_run_pipeline_records_stages(pipeline_project: Path) -> None:
    record = _run(pipeline_project, overrides={"one": {"factor": 3}})
    assert [s.name for s in record.stages] == ["one", "two"]
    one, two = record.stages
    assert one.parameters == {"factor": 3}
    assert one.command[0].startswith("python")
    assert one.command[1:] == ["one.py"]
    assert [i.path for i in one.inputs] == ["in.txt"]
    assert [o.path for o in one.outputs] == ["mid.txt"]
    assert (pipeline_project / "mid.txt").read_text() == "63"
    assert [o.path for o in two.outputs] == ["model.bin", "metrics.json"]
    assert record.metrics == {"test": {"value": 63.0}}
    assert one.started_at <= one.ended_at <= two.started_at
    assert record.overrides == {"one": {"factor": 3}}


def test_command_does_not_leak_interpreter_path(pipeline_project: Path) -> None:
    record = _run(pipeline_project)
    for stage in record.stages:
        assert "/" not in stage.command[0]


def test_failing_stage_raises(pipeline_project: Path) -> None:
    (pipeline_project / "two.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(capture.CaptureError, match="exit code 3"):
        _run(pipeline_project)


def test_missing_output_raises(pipeline_project: Path) -> None:
    (pipeline_project / "two.py").write_text("pass\n")
    with pytest.raises(capture.CaptureError, match="did not produce"):
        _run(pipeline_project)


def test_unknown_override_stage_raises(pipeline_project: Path) -> None:
    with pytest.raises(capture.CaptureError, match="unknown stage"):
        _run(pipeline_project, overrides={"nope": {"x": 1}})


def test_git_commit_and_dirty_state(pipeline_project: Path) -> None:
    root = pipeline_project
    assert capture.git_commit(root) is None
    assert capture.git_is_dirty(root, Path("one.py")) is None

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@example.com", "-c", "user.name=t", "add", "one.py", "two.py")
    git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init")
    commit = capture.git_commit(root)
    assert commit
    assert len(commit) == 40
    assert capture.git_is_dirty(root, Path("one.py")) is False
    (root / "one.py").write_text(STAGE_ONE + "\n# edit\n")
    assert capture.git_is_dirty(root, Path("one.py")) is True

    record = _run(root)
    assert {s.git_commit for s in record.stages} == {commit}
    assert [s.git_dirty for s in record.stages] == [True, False]


def test_environment_capture() -> None:
    env = capture.capture_environment()
    assert env.python_version.startswith("3.")
    assert "pydantic" in {name.lower() for name in env.dependencies}
    assert env.hardware["cpu_count"]


def test_build_includes_pipeline(pipeline_project: Path) -> None:
    _run(pipeline_project)
    passport = build_passport(pipeline_project / CONFIG_FILENAME)
    assert [s.name for s in passport.pipeline] == ["one", "two"]
    assert passport.run is not None
    paths = {a.path: a.kind.value for a in passport.artifacts}
    assert paths["one.py"] == "script"
    assert paths["model.bin"] == "model"
    assert {"in.txt", "mid.txt", "metrics.json"} <= set(paths)
    assert passport.metrics["test"]["value"] == 42.0
    assert passport.metrics["test"]["accuracy"] == 0.91  # manual metrics still merged in


def test_build_requires_run_when_stages_declared(pipeline_project: Path) -> None:
    with pytest.raises(BuildError, match="passport run"):
        build_passport(pipeline_project / CONFIG_FILENAME)


def test_build_rejects_stale_stage_config(pipeline_project: Path) -> None:
    _run(pipeline_project)
    config = yaml.safe_load((pipeline_project / CONFIG_FILENAME).read_text())
    config["stages"][0]["params"]["factor"] = 5
    (pipeline_project / CONFIG_FILENAME).write_text(yaml.safe_dump(config))
    with pytest.raises(BuildError, match="changed since the last"):
        build_passport(pipeline_project / CONFIG_FILENAME)


def test_build_rejects_output_modified_after_run(pipeline_project: Path) -> None:
    _run(pipeline_project)
    (pipeline_project / "mid.txt").write_text("999")
    with pytest.raises(BuildError, match=r"mid\.txt changed since"):
        build_passport(pipeline_project / CONFIG_FILENAME)


def test_runtime_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(capture.PARAMS_ENV, raising=False)
    assert params({"a": 1}) == {"a": 1}
    monkeypatch.setenv(capture.PARAMS_ENV, json.dumps({"a": 2, "b": 3}))
    assert params({"a": 1}) == {"a": 2, "b": 3}


def test_resolve_tracking_uri(tmp_path: Path) -> None:
    assert resolve_tracking_uri("http://localhost:5000", tmp_path) == "http://localhost:5000"
    assert resolve_tracking_uri("mlruns", tmp_path) == (tmp_path / "mlruns").resolve().as_uri()


def test_dvc_add_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dvc_add(tmp_path, [])  # nothing to track is a no-op
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(TrackingUnavailableError, match="not installed"):
        dvc_add(tmp_path, [Path("x")])


def test_dvc_add_invokes_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".dvc").mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/dvc")

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dvc_add(tmp_path, [Path("data/train.csv")])
    assert calls == [["/usr/bin/dvc", "add", "data/train.csv"]]


def test_mlflow_tracker_logs_stages(pipeline_project: Path) -> None:
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    tracking = TrackingConfig(mlflow_uri="sqlite:///" + str(pipeline_project / "mlflow.db"))
    tracker = MlflowTracker(tracking, pipeline_project, "demo")
    parent = tracker.start("abc123")
    config = load_config(pipeline_project / CONFIG_FILENAME)
    record = capture.run_pipeline(
        pipeline_project,
        config,
        on_stage=tracker.log_stage,
    )
    tracker.finish()

    client = MlflowClient(tracking_uri=tracking.mlflow_uri)
    children = client.search_runs(
        [tracker.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent}'"
    )
    assert sorted(r.data.tags["mlflow.runName"] for r in children) == ["one", "two"]
    one = next(r for r in children if r.data.tags["mlflow.runName"] == "one")
    assert one.data.params == {"factor": "2"}
    assert client.get_run(parent).data.metrics["test_value"] == 42.0
    assert len(record.stages) == 2


def test_mlflow_tracker_requires_uri(tmp_path: Path) -> None:
    with pytest.raises(TrackingUnavailableError):
        MlflowTracker(TrackingConfig(), tmp_path, "demo")
