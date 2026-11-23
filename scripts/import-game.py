"""Import a standalone HTTP game build. Run from the personal site repository."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GAME = "return-of-the-king"
RELEASES = ROOT / "public" / GAME / "releases"
PIN = ROOT / "game" / "release.json"
BASE = f"/{GAME}/releases/"
PLACEHOLDER = "__GAME_RELEASE__"


def files(root):
    return sorted(path for path in root.rglob("*") if path.is_file())


def digest(root, release=PLACEHOLDER):
    checksum = hashlib.sha256()
    for path in files(root):
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if name == "index.html":
            actual = f'<base href="{BASE}{release}/">'.encode()
            if data.count(actual) != 1:
                raise ValueError("Missing unique release base")
            data = data.replace(actual, f'<base href="{BASE}{PLACEHOLDER}/">'.encode())
        checksum.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return checksum.hexdigest()


def duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ]))


def import_game(source):
    RELEASES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RELEASES.parent) as tmp:
        stage = Path(tmp) / "dist"
        shutil.copytree(source, stage)
        replacement = {}
        for master in (stage / "assets/music").rglob("*.flac"):
            delivery = master.with_suffix(".m4a")
            subprocess.run([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(master),
                "-map", "0:a:0", "-map_metadata", "-1", "-c:a", "aac",
                "-b:a", "192k", "-movflags", "+faststart", str(delivery),
            ], check=True)
            if abs(duration(master) - duration(delivery)) > 0.03:
                raise ValueError(f"Music duration changed for {master.name}")
            replacement[master.relative_to(stage).as_posix()] = delivery.relative_to(stage).as_posix()
            master.unlink()

        def web_urls(value):
            if isinstance(value, str):
                return replacement.get(value, value)
            if isinstance(value, list):
                return [web_urls(item) for item in value]
            if isinstance(value, dict):
                return {key: web_urls(item) for key, item in value.items()}
            return value

        for manifest in stage.rglob("*.json"):
            original = json.loads(manifest.read_text())
            updated = web_urls(original)
            if updated != original:
                manifest.write_text(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
        index = stage / "index.html"
        html = index.read_text()
        inline = re.findall(r"<script>(.*?)</script>", html, re.S)
        if len(inline) != 1 or html.count("<head>") != 1:
            raise ValueError("Inspect the updated game bootstrap before publishing")
        # Bind the title controls and start fetching the lead before the engine runs.
        bootstrap = stage / "app/title-bootstrap.js"
        pieces = [stage / "app/title-audio.js", stage / "app/arrival.js"]
        bootstrap.write_text("\n;".join([inline[0], *(part.read_text() for part in pieces)]))
        for part in pieces:
            part.unlink()
        html = html.replace("<script>" + inline[0] + "</script>", "")
        campaign = json.loads((stage / "assets/campaigns.json").read_text())["levels"][0]
        cue = campaign["titleMusic"]
        cast = json.loads((stage / "assets/characters_manifest.json").read_text())["models"]
        hero = cast[0]["url"] + ".packed"
        if not (stage / hero).is_file():
            raise ValueError(f"Missing packed title character {hero}")
        end = html.index("</section>", html.index('<section id="story-title"')) + len("</section>")
        audio = (
            f'<audio id="arrival-score" preload="auto" src="./{cue["path"]}"></audio>'
            f'<script src="./app/title-bootstrap.js" data-hero="./{hero}" data-track="./{cue["path"]}" '
            f'data-loop-start="{cue["loopStart"]}" data-loop-end="{cue["loopEnd"]}"></script>'
        )
        html = html[:end] + "\n  " + audio + html[end:]
        html = html.replace("<head>", f'''<head>
  <base href="{BASE}{PLACEHOLDER}/">
  <link rel="preload" as="script" href="./app/title-bootstrap.js">''')
        index.write_text(html)
        sha = digest(stage)
        release = sha[:20]
        index.write_text(html.replace(PLACEHOLDER, release))
        dest = RELEASES / release
        if dest.exists():
            if digest(dest, release) != sha:
                raise ValueError("Existing release contents differ")
        else:
            stage.rename(dest)
        PIN.write_text(json.dumps({"release": release, "sha256": sha}, indent=2) + "\n")
        print(f"Imported {release} ({len(files(dest))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, nargs="?")
    args = parser.parse_args()
    if args.dist:
        import_game(args.dist)
    pin = json.loads(PIN.read_text())
    if digest(RELEASES / pin["release"], pin["release"]) != pin["sha256"]:
        raise SystemExit("Game release integrity check failed")
    print("Verified game release", pin["release"])
