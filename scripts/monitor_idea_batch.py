"""Monitor batch_idea_workflow progress; auto-restart if batch process dies with quota errors."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IDEA_ROOT = REPO / "idea"
BATCH_SCRIPT = REPO / "batch_idea_workflow.py"
BATCH_PID_FILE = REPO / "scripts" / "batch_monitor.pid"

PRIMARY_KEY = "sk-36bWqzTxOqCP1HFFf03fFdoktSGItMZUZyxJcZHMTMcOp4rq"
BACKUP_KEY  = "sk-2gHrK5lYydj4mRl9PqP1NA6Qd9r4RSYPMZYHGRSVJb600HyI"

TARGET_ROUNDS = 10  # per workspace

POLL_INTERVAL = 90  # seconds


def idea_versions(ws: Path) -> list[int]:
    return sorted(int(f.stem.replace("idea_v", "")) for f in ws.glob("idea_v*.md")
                  if f.stem.replace("idea_v", "").isdigit())


def review_versions(ws: Path) -> list[int]:
    return sorted(int(f.stem.replace("idea_review_v", "")) for f in ws.glob("idea_review_v*.md")
                  if f.stem.replace("idea_review_v", "").isdigit())


def workspaces() -> list[Path]:
    return sorted(d for d in IDEA_ROOT.iterdir() if d.is_dir() and (d / "idea_v1.md").exists())


def snapshot() -> dict[str, dict]:
    result = {}
    for ws in workspaces():
        iv = idea_versions(ws)
        rv = review_versions(ws)
        max_idea = max(iv) if iv else 0
        max_rev = max(rv) if rv else 0
        done = max_idea >= (TARGET_ROUNDS + 1) and max_rev >= (TARGET_ROUNDS + 1)
        result[ws.name] = {"ws": ws, "max_idea": max_idea, "max_rev": max_rev, "done": done}
    return result


def batch_running() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if p.info["name"] and "python" in p.info["name"].lower():
                    cmd = " ".join(p.info["cmdline"] or [])
                    if "batch_idea_workflow" in cmd:
                        return True
            except Exception:
                pass
    except ImportError:
        pass
    return False


def start_batch(start_v: int = 1) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u", str(BATCH_SCRIPT),
        "--idea-root", "idea",
        "--rounds", str(TARGET_ROUNDS),
        "--start-version", str(start_v),
        "--timeout-seconds", "240",
        "--retry-limit", "5",
        "--retry-sleep-seconds", "15",
        "--api-key", PRIMARY_KEY,
    ]
    log_path = IDEA_ROOT / "batch_monitor_restart.log"
    log_handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(REPO), stdout=log_handle, stderr=log_handle, text=True)


def active_log_tail(ws_name: str, lines: int = 4) -> str:
    ws = IDEA_ROOT / ws_name
    logs = sorted(ws.glob("iter_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not logs:
        return "  (no log)"
    tail = logs[0].read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return "\n".join(f"  {l}" for l in tail[-lines:])


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    (REPO / "scripts" / "batch_monitor.log").open("a", encoding="utf-8").write(line + "\n")


def main() -> None:
    log("=== monitor started ===")
    last_snap: dict[str, dict] = {}

    while True:
        snap = snapshot()
        all_done = all(v["done"] for v in snap.values())

        # Print status table
        lines = ["Progress snapshot:"]
        for name, info in snap.items():
            status = "DONE" if info["done"] else "..."
            lines.append(f"  {name:12s}  idea=V{info['max_idea']}  review=V{info['max_rev']}  {status}")
        log("\n".join(lines))

        # Report changes
        for name, info in snap.items():
            prev = last_snap.get(name)
            if prev and (info["max_idea"] != prev["max_idea"] or info["max_rev"] != prev["max_rev"]):
                log(f"  [{name}] progress: idea V{prev['max_idea']}->V{info['max_idea']}  review V{prev['max_rev']}->V{info['max_rev']}")

        last_snap = snap

        if all_done:
            log("All workspaces complete!")
            break

        # Show tail of the currently-active workspace log
        active = [n for n, v in snap.items() if not v["done"] and v["max_idea"] > 0]
        if active:
            log(f"Active workspace: {active[0]}\n{active_log_tail(active[0])}")

        # Check if batch is still running; if not, restart it
        if not all_done and not batch_running():
            log("WARNING: batch process not found. Restarting...")
            start_batch()
            time.sleep(10)

        time.sleep(POLL_INTERVAL)

    log("=== monitor done ===")


if __name__ == "__main__":
    main()
