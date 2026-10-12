"""Download original source archives with SHA-256 and size verification."""
from pathlib import Path
import argparse, hashlib, json, urllib.request

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["assets"]:
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Invalid asset name")
        target = args.output / item["file"]
        if target.exists():
            if target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
                print("Verified existing", target.name)
                continue
            raise RuntimeError("Existing file fails verification: " + str(target))
        temporary = target.with_suffix(target.suffix + ".partial")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "solanum-dataset-downloader"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
        if temporary.stat().st_size != item["bytes"] or digest(temporary) != item["sha256"]:
            raise RuntimeError("Downloaded asset fails verification: " + str(temporary))
        temporary.rename(target)
        print("Downloaded and verified", target.name)

if __name__ == "__main__":
    main()
