from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_DIR = WORKSPACE_ROOT / "automation"
EXPERIMENT_LOG_DIR = WORKSPACE_ROOT / "experiment_logs"
STATE_PATH = EXPERIMENT_LOG_DIR / "night_4gpu_watcher_state.json"
LOCAL_LOG_PATH = EXPERIMENT_LOG_DIR / "night_4gpu_watcher.log"

REMOTE_PROJECT_ROOT = "/root/autodl-tmp/BRA_Project"
REMOTE_AUTOMATION_DIR = f"{REMOTE_PROJECT_ROOT}/logs/automation"
REMOTE_LOG_PATH = f"{REMOTE_AUTOMATION_DIR}/night_4gpu_watcher.log"
REMOTE_STATE_PATH = f"{REMOTE_AUTOMATION_DIR}/night_4gpu_watcher_state.json"

SSH_HOST = os.environ.get("NIGHT4GPU_SSH_HOST", "connect.westc.seetacloud.com")
SSH_PORT = int(os.environ.get("NIGHT4GPU_SSH_PORT", "47559"))
SSH_USER = os.environ.get("NIGHT4GPU_SSH_USER", "root")
SSH_PASSWORD = os.environ.get("NIGHT4GPU_SSH_PASSWORD", "aMNIL2fW6aoV")
SSH_KEY_PATH = Path(
    os.environ.get("NIGHT4GPU_SSH_KEY", r"C:\Users\shers\.ssh\id_ed25519_autodl")
)

CHECKLIST_RE = re.compile(r"^今夜4GPU任务清单_[Vv](\d+)\.md$")
COMMAND_DOC_GLOB = "今夜4GPU启动命令*.md"
MARKDOWN_LINK_RE = re.compile(r"\(([^)]+\.md)\)")
CODE_BLOCK_RE = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

DOCVQA_HELPER = WORKSPACE_ROOT / "resume_docvqa_download.py"
VIDEOMME_HELPER = WORKSPACE_ROOT / "launch_videomme_download.py"


@dataclass
class ChecklistDoc:
    version: int
    path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_local_log(message: str) -> None:
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}"
    with LOCAL_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "last_processed_version": 0,
            "last_processed_file": None,
            "last_command_doc": None,
            "last_run_at": None,
            "last_result": None,
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_processed_version": 0,
            "last_processed_file": None,
            "last_command_doc": None,
            "last_run_at": None,
            "last_result": "state_corrupt",
        }


def save_state(state: dict) -> None:
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_checklists() -> Iterable[ChecklistDoc]:
    for path in WORKSPACE_ROOT.iterdir():
        if not path.is_file():
            continue
        match = CHECKLIST_RE.match(path.name)
        if not match:
            continue
        yield ChecklistDoc(version=int(match.group(1)), path=path)


def latest_checklist() -> Optional[ChecklistDoc]:
    docs = sorted(iter_checklists(), key=lambda item: (item.version, item.path.stat().st_mtime))
    return docs[-1] if docs else None


def referenced_markdown_paths(markdown_path: Path) -> list[Path]:
    content = markdown_path.read_text(encoding="utf-8", errors="replace")
    resolved: list[Path] = []
    for raw in MARKDOWN_LINK_RE.findall(content):
        candidate = (markdown_path.parent / raw).resolve()
        if candidate.exists() and candidate.suffix.lower() == ".md":
            resolved.append(candidate)
    return resolved


def pick_command_doc(checklist: ChecklistDoc) -> Optional[Path]:
    referenced = referenced_markdown_paths(checklist.path)
    candidates = [p for p in referenced if "启动命令" in p.name]
    candidates.extend(sorted(WORKSPACE_ROOT.glob(COMMAND_DOC_GLOB)))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    if not unique:
        return None

    def score(path: Path) -> tuple[int, float]:
        text = path.read_text(encoding="utf-8", errors="replace")
        exact_version = f"V{checklist.version}" in path.name or f"V{checklist.version}" in text
        referenced_bonus = 1 if path in referenced else 0
        return (
            100 if exact_version else 0,
            10 if referenced_bonus else 0,
            path.stat().st_mtime,
        )

    return sorted(unique, key=score)[-1]


def extract_launch_blocks(command_doc: Path) -> list[str]:
    content = command_doc.read_text(encoding="utf-8", errors="replace")
    blocks: list[str] = []
    for match in CODE_BLOCK_RE.finditer(content):
        block = match.group(1).strip()
        lower = block.lower()
        if "tail -f" in lower or "ls -lah" in lower or lower.startswith("ssh -p "):
            continue
        if "cuda_visible_devices=" in lower or "nohup bash -lc" in lower:
            blocks.append(block)
    return blocks


def ssh_base_args() -> list[str]:
    args = [
        "ssh",
        "-p",
        str(SSH_PORT),
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if SSH_KEY_PATH.exists():
        args.extend(["-i", str(SSH_KEY_PATH)])
    return args


def scp_base_args() -> list[str]:
    args = [
        "scp",
        "-P",
        str(SSH_PORT),
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if SSH_KEY_PATH.exists():
        args.extend(["-i", str(SSH_KEY_PATH)])
    return args


def run_remote_shell(script: str, timeout: int = 1200) -> tuple[int, str, str]:
    wrapped = "\n".join(
        [
            "source /etc/network_turbo >/dev/null 2>&1 || true",
            'export PATH="/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"',
            'export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"',
            f"cd {shlex.quote(REMOTE_PROJECT_ROOT)}",
            script.strip(),
        ]
    )
    cmd = ssh_base_args() + [f"{SSH_USER}@{SSH_HOST}", f"bash -lc {shlex.quote(wrapped)}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def remote_exists(path: str) -> bool:
    code, _, _ = run_remote_shell(f"test -e {shlex.quote(path)}", timeout=60)
    return code == 0


def remote_process_running(needle: str) -> bool:
    code, out, _ = run_remote_shell(
        f"ps -eo pid,cmd | grep {shlex.quote(needle)} | grep -v grep || true",
        timeout=60,
    )
    return code == 0 and bool(out.strip())


def maybe_resume_downloads(checklist_text: str) -> None:
    lower = checklist_text.lower()
    if "docvqa" in lower and not remote_exists(f"{REMOTE_PROJECT_ROOT}/datasets/DocVQA_hf"):
        if DOCVQA_HELPER.exists():
            append_local_log("DocVQA missing; starting local resume helper.")
            subprocess.run([sys.executable, str(DOCVQA_HELPER)], cwd=str(WORKSPACE_ROOT), check=False)
        else:
            append_local_log("DocVQA missing, but resume helper script is absent.")

    videomme_needed = "video-mme" in lower or "video_mme" in lower or "videomme" in lower
    if videomme_needed and not remote_process_running("_resume_videomme.py"):
        if VIDEOMME_HELPER.exists():
            append_local_log("Video-MME requested; ensuring mirror resume helper is running.")
            subprocess.run([sys.executable, str(VIDEOMME_HELPER)], cwd=str(WORKSPACE_ROOT), check=False)
        else:
            append_local_log("Video-MME requested, but resume helper script is absent.")


def mirror_to_remote() -> None:
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_remote_shell(f"mkdir -p {shlex.quote(REMOTE_AUTOMATION_DIR)}", timeout=60)
    if LOCAL_LOG_PATH.exists():
        subprocess.run(
            scp_base_args() + [str(LOCAL_LOG_PATH), f"{SSH_USER}@{SSH_HOST}:{REMOTE_LOG_PATH}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if STATE_PATH.exists():
        subprocess.run(
            scp_base_args() + [str(STATE_PATH), f"{SSH_USER}@{SSH_HOST}:{REMOTE_STATE_PATH}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def launch_blocks(blocks: list[str]) -> list[dict]:
    results: list[dict] = []
    for idx, block in enumerate(blocks, start=1):
        first_line = next((line.strip() for line in block.splitlines() if line.strip()), "")
        append_local_log(f"Launching block {idx}/{len(blocks)}: {first_line[:120]}")
        code, out, err = run_remote_shell(block, timeout=1800)
        result = {
            "index": idx,
            "exit_code": code,
            "first_line": first_line,
            "stdout_tail": out[-1200:],
            "stderr_tail": err[-1200:],
        }
        if code == 0:
            append_local_log(f"Block {idx} launched successfully.")
        else:
            append_local_log(f"Block {idx} failed with exit code {code}.")
        results.append(result)
    return results


def bootstrap_current_version() -> int:
    state = load_state()
    checklist = latest_checklist()
    if checklist is None:
        append_local_log("Bootstrap requested, but no checklist was found.")
        return 0
    state.update(
        {
            "last_processed_version": checklist.version,
            "last_processed_file": str(checklist.path),
            "last_command_doc": None,
            "last_run_at": now_iso(),
            "last_result": "bootstrapped",
            "last_launch_results": [],
        }
    )
    save_state(state)
    append_local_log(f"Bootstrapped watcher state at {checklist.path.name} (V{checklist.version}).")
    try:
        mirror_to_remote()
    except Exception as exc:
        append_local_log(f"Remote mirror skipped during bootstrap: {exc}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--bootstrap":
        return bootstrap_current_version()

    append_local_log("Watcher tick started.")
    state = load_state()
    checklist = latest_checklist()
    if checklist is None:
        append_local_log("No checklist matching 今夜4GPU任务清单_Vx.md was found.")
        state["last_run_at"] = now_iso()
        state["last_result"] = "no_checklist"
        save_state(state)
        return 0

    append_local_log(f"Latest checklist detected: {checklist.path.name} (V{checklist.version}).")
    if checklist.version <= int(state.get("last_processed_version", 0)):
        append_local_log("No newer checklist version found; nothing to do.")
        state["last_run_at"] = now_iso()
        state["last_result"] = "no_new_version"
        state["last_launch_results"] = []
        save_state(state)
        try:
            mirror_to_remote()
        except Exception as exc:
            append_local_log(f"Remote mirror skipped after no-op tick: {exc}")
        return 0

    checklist_text = checklist.path.read_text(encoding="utf-8", errors="replace")
    command_doc = pick_command_doc(checklist)
    if command_doc is None:
        append_local_log("A new checklist exists, but no matching 启动命令 markdown was found.")
        state["last_run_at"] = now_iso()
        state["last_result"] = "missing_command_doc"
        save_state(state)
        return 1

    append_local_log(f"Selected command document: {command_doc.name}")
    blocks = extract_launch_blocks(command_doc)
    if not blocks:
        append_local_log("Command document was found, but no executable launch blocks were extracted.")
        state["last_run_at"] = now_iso()
        state["last_result"] = "no_launch_blocks"
        state["last_command_doc"] = str(command_doc)
        save_state(state)
        return 1

    try:
        append_local_log("Starting remote preflight.")
        maybe_resume_downloads(checklist_text + "\n" + command_doc.read_text(encoding="utf-8", errors="replace"))
        launch_results = launch_blocks(blocks)
        state.update(
            {
                "last_processed_version": checklist.version,
                "last_processed_file": str(checklist.path),
                "last_command_doc": str(command_doc),
                "last_run_at": now_iso(),
                "last_result": "executed",
                "last_launch_results": launch_results,
            }
        )
        save_state(state)
        mirror_to_remote()
        append_local_log(f"New checklist V{checklist.version} processed successfully.")
        return 0
    except Exception as exc:
        append_local_log(f"Watcher failed: {exc}")
        state["last_run_at"] = now_iso()
        state["last_result"] = f"failed: {exc}"
        save_state(state)
        try:
            mirror_to_remote()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
