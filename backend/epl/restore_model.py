"""Rebuild serving artifacts from sealed reports without rerunning model selection."""
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from .audit import load_audited
from .data import ROOT
from .experiments import digest, make_protocol, publish


def restore(directory=None):
    directory = Path(directory or ROOT / "artifacts/research-v2").resolve()
    if not directory.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Reports must be inside this repository's artifacts directory")
    matches = load_audited()
    audit = json.loads((ROOT / "data/audited/audit.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol != make_protocol(matches, audit):
        raise ValueError("Code, dependency version or audited data differs from the sealed experiment")
    decision = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "confirmation.json").read_text(encoding="utf-8"))
    if decision["protocol_sha256"] != digest(protocol) or result["protocol_sha256"] != digest(protocol):
        raise ValueError("Sealed reports do not refer to this protocol")
    with threadpool_limits(limits=1):
        publish(matches, directory, decision, result, audit)
    return {"active": result["active"], "promoted": result["promoted"], "data_through": audit["data_through"]}


if __name__ == "__main__":
    print(json.dumps(restore(), indent=2))
