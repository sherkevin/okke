from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IDEA_ROOT = "idea"
DEFAULT_SEED_NAME = "idea_v1.md"
DEFAULT_IDEA_PATTERN = "idea_v{version}.md"
DEFAULT_REVIEW_PATTERN = "idea_review_v{version}.md"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RETRY_LIMIT = 4
DEFAULT_RETRY_SLEEP_SECONDS = 20


@dataclass
class WorkspaceStatus:
    slug: str
    workspace: Path
    complete: bool
    last_idea_version: int
    last_review_version: int
    attempts: int
    last_error: str = ""


def discover_workspaces(root: Path, seed_name: str) -> list[Path]:
    return sorted(
        child for child in root.iterdir() if child.is_dir() and (child / seed_name).is_file()
    )


def existing_versions(workspace: Path, prefix: str) -> list[int]:
    versions: list[int] = []
    for child in workspace.iterdir():
        if not child.is_file():
            continue
        name = child.name.lower()
        if not name.startswith(prefix) or not name.endswith(".md"):
            continue
        suffix = name[len(prefix) : -3]
        if suffix.isdigit():
            versions.append(int(suffix))
    return sorted(versions)


def current_status(workspace: Path, max_version: int) -> WorkspaceStatus:
    idea_versions = existing_versions(workspace, "idea_v")
    review_versions = existing_versions(workspace, "idea_review_v")
    last_idea = max(idea_versions) if idea_versions else 0
    last_review = max(review_versions) if review_versions else 0
    complete = last_idea >= max_version and last_review >= max_version
    return WorkspaceStatus(
        slug=workspace.name,
        workspace=workspace,
        complete=complete,
        last_idea_version=last_idea,
        last_review_version=last_review,
        attempts=0,
    )


def append_banner(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {message} ===\n")


def run_workspace(
    repo_root: Path,
    workspace: Path,
    seed_name: str,
    start_version: int,
    max_version: int,
    timeout_seconds: int,
    api_key: str,
    retry_limit: int,
    retry_sleep_seconds: int,
) -> WorkspaceStatus:
    workflow_script = repo_root / "veb_idea_workflow.py"
    log_path = workspace / f"iter_v{start_version}_to_v{max_version}.log"
    err_path = workspace / f"iter_v{start_version}_to_v{max_version}.err.log"
    status = current_status(workspace, max_version)

    if status.complete:
        return status

    cmd = [
        sys.executable,
        str(workflow_script),
        "--workspace",
        str(workspace),
        "--source",
        seed_name,
        "--idea-pattern",
        DEFAULT_IDEA_PATTERN,
        "--review-pattern",
        DEFAULT_REVIEW_PATTERN,
        "--start-version",
        str(start_version),
        "--max-version",
        str(max_version),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])

    for attempt in range(1, retry_limit + 1):
        status.attempts = attempt
        append_banner(log_path, f"{workspace.name} attempt {attempt} start")
        with log_path.open("a", encoding="utf-8") as log_handle, err_path.open(
            "a", encoding="utf-8"
        ) as err_handle:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=log_handle,
                stderr=err_handle,
                text=True,
            )
        status = current_status(workspace, max_version)
        status.attempts = attempt
        if proc.returncode == 0 and status.complete:
            return status
        status.last_error = f"attempt {attempt} failed with rc={proc.returncode}"
        if attempt < retry_limit:
            time.sleep(retry_sleep_seconds)

    return status


def write_summary(summary_path: Path, statuses: list[WorkspaceStatus], max_version: int) -> None:
    lines = [
        f"# Idea Batch Summary (target V{max_version})",
        "",
        "| Idea | Status | Last Idea | Last Review | Attempts |",
        "|------|--------|-----------|-------------|----------|",
    ]
    for status in statuses:
        state = "done" if status.complete else "incomplete"
        lines.append(
            f"| {status.slug} | {state} | V{status.last_idea_version} | V{status.last_review_version} | {status.attempts} |"
        )
        if status.last_error:
            lines.append(f"| {status.slug} error | {status.last_error} | - | - | - |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run idea reviewer/scientist workflow across idea/<slug> workspaces."
    )
    parser.add_argument("--idea-root", default=DEFAULT_IDEA_ROOT, help="Root folder containing idea subfolders")
    parser.add_argument("--seed-name", default=DEFAULT_SEED_NAME, help="Seed filename inside each workspace")
    parser.add_argument("--start-version", type=int, default=1, help="Start version for each idea line")
    parser.add_argument("--rounds", type=int, default=10, help="Number of scientist iterations per idea")
    parser.add_argument("--only", action="append", default=[], help="Limit execution to one or more workspace slugs")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retry-limit", type=int, default=DEFAULT_RETRY_LIMIT)
    parser.add_argument("--retry-sleep-seconds", type=int, default=DEFAULT_RETRY_SLEEP_SECONDS)
    parser.add_argument("--api-key", default="", help="Override API key passed through to veb_idea_workflow.py")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    idea_root = (repo_root / args.idea_root).resolve()
    if not idea_root.is_dir():
        raise SystemExit(f"idea root not found: {idea_root}")

    workspaces = discover_workspaces(idea_root, args.seed_name)
    if args.only:
        allowed = {slug.strip() for slug in args.only if slug.strip()}
        workspaces = [workspace for workspace in workspaces if workspace.name in allowed]
    if not workspaces:
        raise SystemExit(f"no workspaces found under {idea_root} with seed {args.seed_name}")

    max_version = args.start_version + args.rounds
    statuses: list[WorkspaceStatus] = []
    for workspace in workspaces:
        status = run_workspace(
            repo_root=repo_root,
            workspace=workspace,
            seed_name=args.seed_name,
            start_version=args.start_version,
            max_version=max_version,
            timeout_seconds=args.timeout_seconds,
            api_key=args.api_key,
            retry_limit=args.retry_limit,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        statuses.append(status)
        write_summary(summary_path := idea_root / f"batch_summary_v{args.start_version}_to_v{max_version}.md", statuses, max_version)

    write_summary(summary_path, statuses, max_version)

    incomplete = [status.slug for status in statuses if not status.complete]
    if incomplete:
        print("Incomplete workspaces:", ", ".join(incomplete))
        print(f"Summary: {summary_path}")
        return 1

    print("All idea workspaces completed.")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
