"""Synthetic reports verify restoration boundaries without evaluating football matches."""
import json

import pandas as pd
import pytest

from epl import restore_model as restoring


@pytest.fixture
def reports(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts/research-v2"
    directory.mkdir(parents=True)
    (tmp_path / "data/audited").mkdir(parents=True)
    audit = {"status": "passed", "data_through": "2026-05-24"}
    (tmp_path / "data/audited/audit.json").write_text(json.dumps(audit))
    protocol = {"synthetic_fixture": True}
    decision = {"protocol_sha256": restoring.digest(protocol)}
    result = {"protocol_sha256": restoring.digest(protocol), "active": "baseline", "promoted": False}
    for name, payload in (("protocol", protocol), ("selection", decision), ("confirmation", result)):
        (directory / f"{name}.json").write_text(json.dumps(payload))
    monkeypatch.setattr(restoring, "ROOT", tmp_path)
    monkeypatch.setattr(restoring, "load_audited", lambda: pd.DataFrame())
    monkeypatch.setattr(restoring, "make_protocol", lambda *_: protocol)
    return directory, protocol


def test_restore_uses_existing_decision_without_model_selection(reports, monkeypatch):
    directory, _ = reports
    calls = []
    monkeypatch.setattr(restoring, "publish", lambda *args: calls.append(args))
    result = restoring.restore(directory)
    assert result == {"active": "baseline", "promoted": False, "data_through": "2026-05-24"}
    assert len(calls) == 1
    assert calls[0][3]["promoted"] is False


def test_changed_protocol_prevents_artifact_writes(reports, monkeypatch):
    directory, _ = reports
    monkeypatch.setattr(restoring, "make_protocol", lambda *_: {"changed": True})
    monkeypatch.setattr(restoring, "publish", lambda *_: pytest.fail("Must not publish mismatched code/data"))
    with pytest.raises(ValueError, match="differs from the sealed"):
        restoring.restore(directory)


def test_unlinked_reports_prevent_artifact_writes(reports, monkeypatch):
    directory, _ = reports
    (directory / "selection.json").write_text('{"protocol_sha256": "wrong"}')
    monkeypatch.setattr(restoring, "publish", lambda *_: pytest.fail("Must not publish unlinked reports"))
    with pytest.raises(ValueError, match="do not refer"):
        restoring.restore(directory)


def test_report_directory_cannot_escape_artifacts(reports):
    directory, _ = reports
    with pytest.raises(ValueError, match="inside this repository"):
        restoring.restore(directory.parent.parent / "outside")
