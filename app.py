from __future__ import annotations

import json
import os
import re
import shutil
import smtplib
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import gradio as gr
import yaml
from huggingface_hub import HfApi, hf_hub_download

from researcher_mapper.parsers.cv_parser import parse_cv
from researcher_mapper.pipelines.run_target import run_target


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/tmp/academic_star_map"))
JOBS_PATH = DATA_ROOT / "jobs.json"
RUNS_DIR = DATA_ROOT / "runs"
MAX_CV_BYTES = int(os.getenv("MAX_CV_BYTES", str(20 * 1024 * 1024)))
MAX_WORKERS = max(1, int(os.getenv("MAX_WORKERS", "1")))
RESULT_DELETE_AFTER_DOWNLOAD_DAYS = int(os.getenv("RESULT_DELETE_AFTER_DOWNLOAD_DAYS", "1"))
RESULT_DELETE_IF_UNDOWNLOADED_DAYS = int(os.getenv("RESULT_DELETE_IF_UNDOWNLOADED_DAYS", "7"))

HF_RESULTS_REPO = os.getenv("HF_RESULTS_REPO", "")
HF_DATA_TOKEN = os.getenv("HF_DATA_TOKEN") or os.getenv("HF_TOKEN")
ADMIN_CALIBRATION_TOKEN = os.getenv("ADMIN_CALIBRATION_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_lock = threading.Lock()
_api = HfApi()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_content() -> dict[str, str]:
    defaults = {
        "brand": "Academic Star Map",
        "tagline": "",
        "intro_md": "",
        "privacy_note_md": "",
        "admin_note_md": "",
        "email_subject": "Your Academic Star Map is ready",
        "email_footer": "Academic Star Map",
    }
    path = Path("content/site.yaml")
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        defaults.update({k: str(v) for k, v in loaded.items() if v is not None})
    return defaults


CONTENT = _load_content()


def _read_jobs() -> dict[str, dict[str, Any]]:
    if not JOBS_PATH.exists():
        return {}
    try:
        with JOBS_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOBS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2, ensure_ascii=False)
    tmp.replace(JOBS_PATH)


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _lock:
        jobs = _read_jobs()
        record = jobs.setdefault(job_id, {"job_id": job_id})
        record.update(updates)
        record["updated_at"] = _iso()
        _write_jobs(jobs)
        _upload_metadata(job_id, record)
        return record


def _get_job(job_id: str) -> dict[str, Any] | None:
    job_id = (job_id or "").strip()
    if not job_id:
        return None
    with _lock:
        local = _read_jobs().get(job_id)
    if local:
        return local
    return _download_metadata(job_id)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    return cleaned[:90] or "cv"


def _job_dir(job_id: str) -> Path:
    return RUNS_DIR / job_id


def _result_zip_path(job_id: str) -> Path:
    return _job_dir(job_id) / f"academic-star-map-{job_id}.zip"


def _repo_enabled() -> bool:
    return bool(HF_RESULTS_REPO and HF_DATA_TOKEN)


def _upload_file(path: Path, path_in_repo: str) -> str | None:
    if not _repo_enabled() or not path.exists():
        return None
    try:
        _api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
            token=HF_DATA_TOKEN,
        )
        return path_in_repo
    except Exception:
        return None


def _upload_metadata(job_id: str, record: dict[str, Any]) -> None:
    if not _repo_enabled():
        return
    meta_path = _job_dir(job_id) / "job.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    public_record = dict(record)
    public_record.pop("email", None)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(public_record, fh, indent=2, ensure_ascii=False)
    _upload_file(meta_path, f"jobs/{job_id}/job.json")


def _download_metadata(job_id: str) -> dict[str, Any] | None:
    if not _repo_enabled():
        return None
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=f"jobs/{job_id}/job.json",
            repo_type="dataset",
            token=HF_DATA_TOKEN,
        )
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _download_result_from_repo(job_id: str, path_in_repo: str) -> Path | None:
    if not _repo_enabled() or not path_in_repo:
        return None
    try:
        local = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            token=HF_DATA_TOKEN,
        )
        target = _result_zip_path(job_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
        return target
    except Exception:
        return None


def _delete_repo_file(path_in_repo: str | None) -> None:
    if not _repo_enabled() or not path_in_repo:
        return
    try:
        _api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
            token=HF_DATA_TOKEN,
        )
    except Exception:
        pass


def _make_download_url(job_id: str) -> str:
    base = PUBLIC_BASE_URL
    if not base:
        host = os.getenv("SPACE_HOST", "").strip()
        if host:
            base = host if host.startswith("http") else f"https://{host}"
    if not base:
        return f"Job ID: {job_id}"
    return f"{base.rstrip('/')}/?job={job_id}"


def _send_ready_email(to_addr: str, job_id: str) -> str | None:
    if not to_addr:
        return "No submitter email was provided."
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    mail_from = os.getenv("MAIL_FROM") or username
    if not all([host, username, password, mail_from]):
        return "SMTP is not configured; no email was sent."

    template_path = Path("content/emails/result_ready.md")
    body = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
    body = body.format(
        brand=CONTENT["brand"],
        job_id=job_id,
        download_url=_make_download_url(job_id),
    )

    msg = EmailMessage()
    msg["Subject"] = CONTENT["email_subject"]
    msg["From"] = mail_from
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(msg)
        return None
    except Exception as exc:
        return f"Could not send email: {exc}"


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def _cleanup_expired_jobs() -> None:
    with _lock:
        jobs = _read_jobs()
    changed = False
    for job_id, record in list(jobs.items()):
        if record.get("status") != "completed":
            continue
        completed_at = _parse_iso(record.get("completed_at"))
        downloaded_at = _parse_iso(record.get("downloaded_at"))
        expires_at: datetime | None = None
        if downloaded_at:
            expires_at = downloaded_at + timedelta(days=RESULT_DELETE_AFTER_DOWNLOAD_DAYS)
        elif completed_at:
            expires_at = completed_at + timedelta(days=RESULT_DELETE_IF_UNDOWNLOADED_DAYS)
        if expires_at and _now() >= expires_at:
            shutil.rmtree(_job_dir(job_id), ignore_errors=True)
            _delete_repo_file(record.get("artifact_repo_path"))
            record["status"] = "expired"
            record["expired_at"] = _iso()
            changed = True
    if changed:
        with _lock:
            _write_jobs(jobs)


def _run_job(
    job_id: str,
    *,
    name: str,
    institution: str,
    orcid: str,
    email: str,
    faculty_page: str,
    cv_path: Path,
    retain_cv: bool,
) -> None:
    try:
        _update_job(job_id, status="running", message="Parsing CV", started_at=_iso())
        cv_data = parse_cv(cv_path)

        if retain_cv:
            retained_path = _upload_file(cv_path, f"calibration/{job_id}/{cv_path.name}")
            _update_job(job_id, calibration_cv_path=retained_path)

        try:
            cv_path.unlink(missing_ok=True)
        except Exception:
            pass

        _update_job(job_id, message="Mapping researchers")
        pipeline_dir = _job_dir(job_id) / "pipeline"
        result = run_target(
            researcher_name=name,
            institution=institution or None,
            orcid=orcid or None,
            faculty_page_url=faculty_page or None,
            cv_data=cv_data,
            output_dir=pipeline_dir,
        )

        graph_path = Path(result.get("graph_path", ""))
        run_dir = graph_path.parent if graph_path else pipeline_dir
        zip_path = _result_zip_path(job_id)
        _zip_directory(run_dir, zip_path)
        artifact_path = _upload_file(zip_path, f"results/{job_id}/{zip_path.name}")

        warning = _send_ready_email(email, job_id)
        _update_job(
            job_id,
            status="completed",
            message="Ready to download",
            completed_at=_iso(),
            result_zip=str(zip_path),
            artifact_repo_path=artifact_path,
            email_warning=warning,
        )
    except Exception as exc:
        try:
            cv_path.unlink(missing_ok=True)
        except Exception:
            pass
        _update_job(job_id, status="failed", message=str(exc), failed_at=_iso())


def start_job(
    name: str,
    institution: str,
    orcid: str,
    submitter_email: str,
    faculty_page: str,
    cv_file: str,
    retain_for_calibration: bool,
    admin_token: str,
) -> tuple[str, str]:
    _cleanup_expired_jobs()
    name = (name or "").strip()
    submitter_email = (submitter_email or "").strip()
    if not name:
        return "Please enter the researcher's full name.", ""
    if not submitter_email:
        return "Please enter an email address so the site can notify you.", ""
    if not cv_file:
        return "Please upload a CV file.", ""

    src = Path(cv_file)
    if not src.exists():
        return "The uploaded CV file could not be read.", ""
    if src.stat().st_size > MAX_CV_BYTES:
        return f"CV is too large. Maximum size is {MAX_CV_BYTES // (1024 * 1024)} MB.", ""
    if src.suffix.lower() not in {".pdf", ".txt"}:
        return "Please upload a PDF or plain-text CV.", ""

    retain_allowed = bool(
        retain_for_calibration
        and ADMIN_CALIBRATION_TOKEN
        and admin_token == ADMIN_CALIBRATION_TOKEN
    )
    if retain_for_calibration and not retain_allowed:
        return "Calibration retention requires the admin token.", ""

    job_id = uuid.uuid4().hex[:12]
    input_dir = _job_dir(job_id) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cv_path = input_dir / _safe_name(src.name)
    shutil.copy2(src, cv_path)

    _update_job(
        job_id,
        status="queued",
        message="Waiting for worker",
        submitted_at=_iso(),
        name=name,
        email=submitter_email,
        retain_cv=retain_allowed,
    )
    _executor.submit(
        _run_job,
        job_id,
        name=name,
        institution=(institution or "").strip(),
        orcid=(orcid or "").strip(),
        email=submitter_email,
        faculty_page=(faculty_page or "").strip(),
        cv_path=cv_path,
        retain_cv=retain_allowed,
    )
    return (
        f"Job `{job_id}` is queued. You can close the page; an email will be sent when the archive is ready.",
        job_id,
    )


def check_status(job_id: str) -> str:
    _cleanup_expired_jobs()
    record = _get_job(job_id)
    if not record:
        return "No job found for that ID."
    status = record.get("status", "unknown")
    message = record.get("message", "")
    lines = [f"**Job:** `{record.get('job_id', job_id)}`", f"**Status:** {status}"]
    if message:
        lines.append(f"**Message:** {message}")
    if status == "completed":
        lines.append("The result archive is ready. Use **Download result** below.")
        if record.get("email_warning"):
            lines.append(f"Email note: {record['email_warning']}")
    if status == "expired":
        lines.append("This result archive has expired and was deleted.")
    return "\n\n".join(lines)


def download_result(job_id: str) -> tuple[str | None, str]:
    _cleanup_expired_jobs()
    record = _get_job(job_id)
    if not record:
        return None, "No job found for that ID."
    if record.get("status") != "completed":
        return None, f"Job is not ready yet. Current status: {record.get('status')}"

    zip_path = Path(record.get("result_zip") or _result_zip_path(job_id))
    if not zip_path.exists():
        zip_path = _download_result_from_repo(job_id, record.get("artifact_repo_path", "")) or zip_path
    if not zip_path.exists():
        return None, "The result archive is missing. Please contact the site owner."

    if not record.get("downloaded_at"):
        _update_job(job_id, downloaded_at=_iso())
    return str(zip_path), "Download started. This archive will be deleted 1 day after download."


def _build_app() -> gr.Blocks:
    css = """
    body { background: #f8fafc; }
    .asm-wrap { max-width: 980px; margin: 0 auto; }
    """
    with gr.Blocks(title=CONTENT["brand"], css=css) as demo:
        gr.Markdown(
            f"<div class='asm-wrap'><h1>{CONTENT['brand']}</h1>"
            f"<p><strong>{CONTENT['tagline']}</strong></p></div>"
        )
        gr.Markdown(CONTENT["intro_md"])

        with gr.Row():
            with gr.Column(scale=1):
                name = gr.Textbox(label="Researcher full name", placeholder="Daniel Soudry")
                institution = gr.Textbox(label="Institution hint", placeholder="Technion")
                orcid = gr.Textbox(label="ORCID", placeholder="0000-0000-0000-0000")
                submitter_email = gr.Textbox(label="Email for notification")
                faculty_page = gr.Textbox(label="Faculty page URL")
                cv_file = gr.File(
                    label="CV file (PDF or TXT, max 20 MB)",
                    file_types=[".pdf", ".txt"],
                    type="filepath",
                )
                with gr.Accordion("Admin calibration", open=False):
                    gr.Markdown(CONTENT["admin_note_md"])
                    retain_for_calibration = gr.Checkbox(label="Retain this CV for calibration")
                    admin_token = gr.Textbox(label="Admin token", type="password")
                submit = gr.Button("Create star map", variant="primary")
            with gr.Column(scale=1):
                job_id = gr.Textbox(label="Job ID")
                status = gr.Markdown("Submit a CV to start.")
                check = gr.Button("Check status")
                download = gr.Button("Download result")
                result_file = gr.File(label="Result archive")

        submit.click(
            start_job,
            inputs=[
                name,
                institution,
                orcid,
                submitter_email,
                faculty_page,
                cv_file,
                retain_for_calibration,
                admin_token,
            ],
            outputs=[status, job_id],
        )
        check.click(check_status, inputs=[job_id], outputs=[status])
        download.click(download_result, inputs=[job_id], outputs=[result_file, status])

        gr.Markdown(CONTENT["privacy_note_md"])

    return demo


if __name__ == "__main__":
    app = _build_app()
    app.queue(default_concurrency_limit=MAX_WORKERS).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
