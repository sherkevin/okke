from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from chord.anchor_builder import DetectedAnchor


@dataclass
class GroundingDinoClient:
    python_executable: str
    server_script: str
    model_path: str
    device: str

    def __post_init__(self) -> None:
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        self._proc = subprocess.Popen(
            [
                self.python_executable,
                self.server_script,
                "--model-path",
                self.model_path,
                "--device",
                self.device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
        )

    def close(self) -> None:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write("__EXIT__\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                pass
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._proc.stderr is not None:
            self._proc.stderr.close()
        self._proc.wait(timeout=30)

    def detect(
        self,
        *,
        image_path: str,
        query: str,
        box_threshold: float,
        text_threshold: float,
        max_boxes: int,
    ) -> list[DetectedAnchor]:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("GroundingDinoClient pipes are not available")
        request = {
            "image_path": image_path,
            "query": query,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "max_boxes": max_boxes,
        }
        self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr = ""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            raise RuntimeError(f"Grounding DINO server returned no response. stderr={stderr}")
        payload = json.loads(response_line)
        return [
            DetectedAnchor(
                box=tuple(anchor["box"]),
                confidence=float(anchor["confidence"]),
                phrase=anchor.get("phrase"),
            )
            for anchor in payload.get("anchors", [])
        ]

    def __enter__(self) -> "GroundingDinoClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
