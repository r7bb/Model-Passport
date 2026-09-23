"""JSON-LD export using W3C PROV-O, DCAT, and ML Schema terms.

Artifacts are content-addressed (``urn:sha256:<hash>``), so the same dataset used by two
passports resolves to the same node across graphs.
"""

from __future__ import annotations

from typing import Any

from model_passport.core.schema import ArtifactKind, Passport

CONTEXT = {
    "prov": "http://www.w3.org/ns/prov#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "dcterms": "http://purl.org/dc/terms/",
    "mls": "http://www.w3.org/ns/mls#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "mp": "https://github.com/r7bb/Model-Passport/ns#",
    "used": {"@id": "prov:used", "@type": "@id"},
    "wasGeneratedBy": {"@id": "prov:wasGeneratedBy", "@type": "@id"},
    "wasAssociatedWith": {"@id": "prov:wasAssociatedWith", "@type": "@id"},
    "wasDerivedFrom": {"@id": "prov:wasDerivedFrom", "@type": "@id"},
    "startedAtTime": {"@id": "prov:startedAtTime", "@type": "xsd:dateTime"},
    "endedAtTime": {"@id": "prov:endedAtTime", "@type": "xsd:dateTime"},
    "generatedAtTime": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
}

_KIND_TYPES = {
    ArtifactKind.DATASET: ["prov:Entity", "dcat:Dataset"],
    ArtifactKind.MODEL: ["prov:Entity", "mls:Model"],
    ArtifactKind.SCRIPT: ["prov:Entity", "prov:Plan", "mls:Implementation"],
    ArtifactKind.CONFIG: ["prov:Entity", "prov:Plan"],
    ArtifactKind.OTHER: ["prov:Entity"],
}


def _artifact_id(sha256: str) -> str:
    return f"urn:sha256:{sha256}"


def to_jsonld(passport: Passport) -> dict[str, Any]:
    pid = f"urn:uuid:{passport.identity.passport_id}"
    tool = {"@id": "mp:model-passport", "@type": ["prov:Agent", "prov:SoftwareAgent"]}
    owner = passport.declared.owner
    owner_node = (
        {"@id": f"{pid}#owner", "@type": ["prov:Agent"], "dcterms:title": owner} if owner else None
    )
    agents = [tool] + ([owner_node] if owner_node else [])
    agent_ids = [a["@id"] for a in agents]

    graph: list[dict[str, Any]] = list(agents)
    by_path = {a.path: a for a in passport.artifacts}
    for artifact in passport.artifacts:
        graph.append(
            {
                "@id": _artifact_id(artifact.sha256),
                "@type": _KIND_TYPES[artifact.kind],
                "dcterms:identifier": artifact.path,
                "mp:sha256": artifact.sha256,
                "dcat:byteSize": artifact.size_bytes,
            }
        )

    generated_by: dict[str, str] = {}
    for index, stage in enumerate(passport.pipeline):
        activity_id = f"{pid}#stage-{index}-{stage.name}"
        activity: dict[str, Any] = {
            "@id": activity_id,
            "@type": ["prov:Activity", "mls:Run"],
            "dcterms:title": stage.name,
            "used": [_artifact_id(i.sha256) for i in stage.inputs]
            + [_artifact_id(stage.script_sha256)],
            "wasAssociatedWith": agent_ids,
            "mp:parameters": stage.parameters,
            "mp:gitCommit": stage.git_commit,
        }
        if stage.started_at:
            activity["startedAtTime"] = stage.started_at.isoformat()
        if stage.ended_at:
            activity["endedAtTime"] = stage.ended_at.isoformat()
        graph.append(activity)
        for output in stage.outputs:
            generated_by[output.path] = activity_id

    for node in graph:
        path = node.get("dcterms:identifier")
        if path in generated_by and path in by_path:
            node["wasGeneratedBy"] = generated_by[path]

    model = passport.model
    passport_node: dict[str, Any] = {
        "@id": pid,
        "@type": ["prov:Entity", "mp:Passport"],
        "dcterms:title": f"{passport.identity.model_name} {passport.identity.version}",
        "generatedAtTime": passport.identity.created_at.isoformat(),
        "wasAssociatedWith": agent_ids,
        "mp:merkleRoot": passport.identity.merkle_root,
        "mp:publicKeyFingerprint": passport.identity.public_key_fingerprint,
        "wasDerivedFrom": [_artifact_id(a.sha256) for a in passport.artifacts]
        + [f"urn:uuid:{link}" for link in passport.lineage_links],
    }
    if model is not None:
        passport_node["mp:model"] = _artifact_id(model.artifact_sha256)
        passport_node["mls:implements"] = model.algorithm
    if passport.policy is not None:
        passport_node["mp:verdict"] = passport.policy.verdict.value
    if passport.metrics:
        passport_node["mls:hasOutput"] = [
            {"@type": "mls:ModelEvaluation", "mp:split": split, "mls:specifiedBy": name,
             "mls:hasValue": value}
            for split, values in passport.metrics.items()
            for name, value in values.items()
        ]  # fmt: skip
    graph.append(passport_node)
    return {"@context": CONTEXT, "@graph": graph}
