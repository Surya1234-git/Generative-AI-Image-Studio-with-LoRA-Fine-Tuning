"""Background jobs (GAN training, LoRA training, classifier training).

Streamlit re-runs the whole script on every click, so long jobs must not live inside
a script run. Each job runs in its own thread and exposes progress/history that the UI
polls. `exclusive=True` jobs (anything using the GPU) cannot run at the same time.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable


class JobBusyError(RuntimeError):
    pass


class StopRequested(Exception):
    """Raised inside a job's thread (by Job.check) when the user pressed Stop."""


ACTIVE = {"queued", "running", "paused", "stopping"}


class Job:
    def __init__(self, kind: str, name: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.name = kind, name
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.started = time.time()
        self.ended: float | None = None
        self.logs: list[str] = []
        self.history: list[dict] = []
        self.result: Any = None
        self.error: str | None = None
        self.extra: dict = {}
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._ckpt = threading.Event()
        self._lock = threading.Lock()

    # -- called from inside the job thread -------------------------------------------
    def check(self) -> None:
        """Call once per training step: honours Pause and Stop at a safe point."""
        if self._stop.is_set():
            raise StopRequested()
        if self._pause.is_set():
            self._set_status("paused")
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                raise StopRequested()
            self._set_status("running")

    def log(self, text: str) -> None:
        with self._lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")
            self.logs = self.logs[-200:]

    def checkpoint_requested(self) -> bool:
        if self._ckpt.is_set():
            self._ckpt.clear()
            return True
        return False

    # -- called from the UI ----------------------------------------------------------
    def pause(self): self._pause.set()
    def resume(self): self._pause.clear()
    def stop(self):
        self._stop.set(); self._pause.clear(); self._set_status("stopping")
    def request_checkpoint(self): self._ckpt.set()

    def is_active(self) -> bool:
        return self.status in ACTIVE

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    def _set_status(self, status: str) -> None:
        with self._lock:
            if self.status not in ("completed", "failed", "stopped"):
                self.status = status


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, name: str, target: Callable[..., Any], exclusive: bool = True, **kwargs) -> Job:
        with self._lock:
            if exclusive:
                for j in self._jobs.values():
                    if j.is_active() and j.extra.get("exclusive", True):
                        raise JobBusyError(f"'{j.name}' is still {j.status}. Stop it or wait for it to finish first.")
            job = Job(kind, name)
            job.extra["exclusive"] = exclusive
            self._jobs[job.id] = job

        def runner():
            job._set_status("running")
            try:
                job.result = target(job, **kwargs)
                job._set_status("completed")
            except StopRequested:
                job.log("Stopped by user.")
                job._set_status("stopped")
            except Exception as exc:  # report the error, never kill the app
                job.error = f"{type(exc).__name__}: {exc}"
                job.log(traceback.format_exc(limit=3))
                with job._lock:
                    job.status = "failed"
            finally:
                job.ended = time.time()
                if job.status == "stopping":
                    job.status = "stopped"

        threading.Thread(target=runner, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str | None) -> Job | None:
        return self._jobs.get(job_id) if job_id else None

    def active(self, kind: str | None = None) -> Job | None:
        for j in reversed(list(self._jobs.values())):
            if j.is_active() and (kind is None or j.kind == kind):
                return j
        return None

    def latest(self, kind: str) -> Job | None:
        for j in reversed(list(self._jobs.values())):
            if j.kind == kind:
                return j
        return None

    def all(self) -> list[Job]:
        return list(self._jobs.values())


JOBS = JobManager()
