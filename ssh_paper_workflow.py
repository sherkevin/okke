from __future__ import annotations

import argparse
import os
import posixpath
import shlex
from datetime import datetime
from pathlib import Path

import paramiko


HOST = "connect.westd.seetacloud.com"
PORT = 23427
USER = "root"
PASSWORD = "aMNIL2fW6aoV"
REMOTE_BASE = "/root/autodl-tmp/BRA_Project/paper_workflow_runs"

DEFAULT_API_URL = "https://new.lemonapi.site/v1"
DEFAULT_MODEL = "[L]gemini-3.1-pro-preview"

SYNC_PATTERNS = [
    "论文大纲_V1.md",
    "论文大纲_V2.md",
    "Review_Strict_V1.md",
    "Review_Strict_V2.md",
    "Revision_Log_V2.md",
    "Scientist_Memory.md",
    "paper_workflow_state.json",
    "paper_workflow_prompt_strategy.md",
    "Pilot_Comparison.md",
    "Workflow_Scaleup_Guide.md",
]


def ssh_connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    return client


def ssh_run(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


def sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    sftp.put(str(local_path), remote_path)


def download_file_if_exists(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    local_path: Path,
    overwrite: bool = False,
) -> bool:
    try:
        sftp.stat(remote_path)
    except OSError:
        return False
    if local_path.exists() and not overwrite:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    return True


def build_remote_command(
    remote_dir: str,
    api_url: str,
    model: str,
    max_version: int,
    timeout_seconds: int,
    api_key: str,
    backup_api_key: str,
) -> str:
    exports = [
        f"export LEMON_API_KEY={shlex.quote(api_key)}",
    ]
    if backup_api_key:
        exports.append(f"export LEMON_API_KEY_BACKUP={shlex.quote(backup_api_key)}")
    exports_text = " && ".join(exports)
    return (
        "export PATH=\"/root/miniconda3/bin:$PATH\""
        f" && mkdir -p {shlex.quote(remote_dir)}"
        f" && cd {shlex.quote(remote_dir)}"
        f" && {exports_text}"
        " && python3 paper_adversarial_workflow.py"
        f" --workspace {shlex.quote(remote_dir)}"
        " --source 论文大纲.md"
        f" --api-url {shlex.quote(api_url)}"
        f" --model {shlex.quote(model)}"
        f" --timeout-seconds {timeout_seconds}"
        f" --max-version {max_version}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper workflow on the SSH server and sync outputs back.")
    parser.add_argument("--workspace", default=".", help="Local workspace directory")
    parser.add_argument("--source", default="论文大纲.md", help="Local source paper filename")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--max-version", type=int, default=2, help="Maximum paper version to generate")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Single API call timeout on remote workflow")
    parser.add_argument("--api-key", default="", help="Primary API key")
    parser.add_argument("--backup-api-key", default="", help="Backup API key")
    parser.add_argument("--probe-only", action="store_true", help="Only run a quick remote API probe")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    source_path = workspace / args.source
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    api_key = args.api_key or os.environ.get("LEMON_API_KEY", "")
    backup_api_key = args.backup_api_key or os.environ.get("LEMON_API_KEY_BACKUP", "")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set LEMON_API_KEY.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = posixpath.join(REMOTE_BASE, timestamp)
    local_archive_dir = workspace / "paper_workflow_remote_runs" / timestamp
    local_archive_dir.mkdir(parents=True, exist_ok=True)

    client = ssh_connect()
    try:
        sftp = client.open_sftp()
        try:
            sftp_mkdir_p(sftp, remote_dir)
            upload_file(sftp, workspace / "paper_adversarial_workflow.py", posixpath.join(remote_dir, "paper_adversarial_workflow.py"))
            upload_file(sftp, source_path, posixpath.join(remote_dir, "论文大纲.md"))
        finally:
            sftp.close()

        probe_command = (
            "export PATH=\"/root/miniconda3/bin:$PATH\""
            f" && export LEMON_API_KEY={shlex.quote(api_key)}"
            + (f" && export LEMON_API_KEY_BACKUP={shlex.quote(backup_api_key)}" if backup_api_key else "")
            + " && python3 - <<'PY'\n"
            + "import json, os, time, urllib.request\n"
            + "headers={\"Authorization\": f\"Bearer {os.environ['LEMON_API_KEY']}\"}\n"
            + "t=time.time()\n"
            + "req=urllib.request.Request('https://new.lemonapi.site/v1/models', headers=headers)\n"
            + "with urllib.request.urlopen(req, timeout=30) as r:\n"
            + "    body=json.loads(r.read().decode('utf-8'))\n"
            + "print(f\"models_count={len(body.get('data', []))}\")\n"
            + "print(f\"elapsed_sec={time.time()-t:.2f}\")\n"
            + "PY"
        )
        code, probe_out, probe_err = ssh_run(client, probe_command, timeout=90)
        print("=== Remote API Probe ===")
        print(probe_out.strip())
        if probe_err.strip():
            print(probe_err.strip())
        if code != 0:
            raise SystemExit(f"Remote probe failed with exit code {code}")
        if args.probe_only:
            return 0

        run_command = build_remote_command(
            remote_dir=remote_dir,
            api_url=args.api_url,
            model=args.model,
            max_version=args.max_version,
            timeout_seconds=args.timeout_seconds,
            api_key=api_key,
            backup_api_key=backup_api_key,
        )
        print(f"=== Remote Run Dir ===\n{remote_dir}")
        code, run_out, run_err = ssh_run(client, run_command, timeout=max(1800, args.timeout_seconds + 600))
        print("=== Remote Workflow Output ===")
        if run_out.strip():
            print(run_out.strip())
        if run_err.strip():
            print(run_err.strip())
        sftp = client.open_sftp()
        try:
            for filename in SYNC_PATTERNS:
                remote_path = posixpath.join(remote_dir, filename)
                archive_path = local_archive_dir / filename
                downloaded = download_file_if_exists(sftp, remote_path, archive_path, overwrite=True)
                if downloaded:
                    canonical_path = workspace / filename
                    download_file_if_exists(sftp, remote_path, canonical_path, overwrite=not canonical_path.exists())
        finally:
            sftp.close()

        print("=== Sync Complete ===")
        print(f"Remote run dir: {remote_dir}")
        print(f"Local archive dir: {local_archive_dir}")
        if code != 0:
            raise SystemExit(f"Remote workflow failed with exit code {code}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
