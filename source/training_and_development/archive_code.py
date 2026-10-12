"""Preserve source bytes by hash so later edits cannot erase run provenance."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    archive = ROOT / "code_versions"
    archive.mkdir(exist_ok=True)
    sources = list(ROOT.glob("*.py")) + list((ROOT / "external_data").glob("*.py"))
    sources += [ROOT / name for name in ("development_plan.json", "RESEARCH_CHARTER.md", "environment_packages.txt")]
    entries = []
    for path in sorted(sources):
        if not path.exists():
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        target = archive / f"{digest}__{path.name}"
        if target.exists():
            assert target.read_bytes() == payload
        else:
            target.write_bytes(payload)
        entries.append({"source": str(path.relative_to(ROOT)), "sha256": digest, "archive": str(target.relative_to(ROOT))})
    report = {"created_at": datetime.datetime.now().astimezone().isoformat(), "entries": entries,
              "notes": ["Only source/configuration text is archived; no raw user imagery is published or copied here.",
                        "Run config script_sha256 identifies its matching preserved train source."]}
    name = datetime.datetime.now().strftime("manifest_%Y%m%d_%H%M%S_%f.json")
    (archive / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archived_files": len(entries), "manifest": str(archive / name)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
