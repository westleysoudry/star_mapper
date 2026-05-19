from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import smtplib
import tempfile
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

from researcher_mapper.parsers.cv_parser import parse_cv
from researcher_mapper.pipelines.run_target import run_target


DATA_ROOT = Path(os.getenv("DATA_ROOT") or Path(tempfile.gettempdir()) / "academic_star_map")
JOBS_PATH = DATA_ROOT / "jobs.json"
RUNS_DIR = DATA_ROOT / "runs"
MAX_CV_BYTES = int(os.getenv("MAX_CV_BYTES", str(20 * 1024 * 1024)))
MAX_WORKERS = max(1, int(os.getenv("MAX_WORKERS", "1")))
RESULT_DELETE_AFTER_DOWNLOAD_DAYS = int(os.getenv("RESULT_DELETE_AFTER_DOWNLOAD_DAYS", "1"))
RESULT_DELETE_IF_UNDOWNLOADED_DAYS = int(os.getenv("RESULT_DELETE_IF_UNDOWNLOADED_DAYS", "7"))

HF_RESULTS_REPO = os.getenv("HF_RESULTS_REPO", "")
HF_DATA_TOKEN = os.getenv("HF_DATA_TOKEN") or os.getenv("HF_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
ADMIN_COPY_ENABLED = os.getenv("ADMIN_COPY_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADMIN_COPY_EMAIL = os.getenv("ADMIN_COPY_EMAIL", "westleysoudry@gmail.com").strip()
EMAIL_BACKEND = os.getenv("EMAIL_BACKEND", "brevo").strip().lower()

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


def _result_starmap_path(job_id: str) -> Path:
    return _job_dir(job_id) / f"academic-star-map-{job_id}.html"


def _result_excel_path(job_id: str) -> Path:
    return _job_dir(job_id) / f"academic-star-map-{job_id}.xlsx"


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


def _download_file_from_repo(path_in_repo: str, target: Path) -> Path | None:
    if not _repo_enabled() or not path_in_repo:
        return None
    try:
        local = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            token=HF_DATA_TOKEN,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
        return target
    except Exception:
        return None


def _download_result_from_repo(job_id: str, path_in_repo: str) -> Path | None:
    return _download_file_from_repo(path_in_repo, _result_zip_path(job_id))


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


def _attachment_payloads(paths: list[Path]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        payloads.append(
            {
                "name": path.name,
                "content": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return payloads


def _send_brevo_email(
    to_addr: str,
    *,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> str | None:
    api_key = os.getenv("BREVO_API_KEY")
    if not api_key:
        return "Brevo API is not configured."
    mail_from = os.getenv("MAIL_FROM") or os.getenv("SMTP_USERNAME")
    if not mail_from:
        return "MAIL_FROM is required for Brevo email delivery."

    payload = {
        "sender": {
            "email": mail_from,
            "name": os.getenv("MAIL_FROM_NAME") or CONTENT["brand"],
        },
        "to": [{"email": to_addr}],
        "subject": subject,
        "textContent": body,
    }
    attachment_payloads = _attachment_payloads(attachments or [])
    if attachment_payloads:
        payload["attachment"] = attachment_payloads
    try:
        response = httpx.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "accept": "application/json",
                "api-key": api_key,
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return None
    except Exception as exc:
        return f"Could not send email with Brevo API: {exc}"


def _send_smtp_email(
    to_addr: str,
    *,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> str | None:
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    mail_from = os.getenv("MAIL_FROM") or username
    if not all([host, username, password, mail_from]):
        return "Email is not configured; no email was sent."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = to_addr
    msg.set_content(body)
    for path in attachments or []:
        if not path.exists() or not path.is_file():
            continue
        content_type, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (content_type or "application/octet-stream").split("/", 1)
        msg.add_attachment(
            path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(msg)
        return None
    except Exception as exc:
        return f"Could not send email: {exc}"


def _send_email(
    to_addr: str,
    *,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> str | None:
    if not to_addr:
        return "No submitter email was provided."
    if os.getenv("BREVO_API_KEY"):
        return _send_brevo_email(
            to_addr,
            subject=subject,
            body=body,
            attachments=attachments,
        )
    if EMAIL_BACKEND != "smtp":
        return "Email API is not configured; add BREVO_API_KEY to the Space secrets."
    return _send_smtp_email(
        to_addr,
        subject=subject,
        body=body,
        attachments=attachments,
    )


def _send_ready_email(to_addr: str, job_id: str) -> str | None:
    template_path = Path("content/emails/result_ready.md")
    body = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
    body = body.format(
        brand=CONTENT["brand"],
        job_id=job_id,
        download_url=_make_download_url(job_id),
    )
    return _send_email(to_addr, subject=CONTENT["email_subject"], body=body)


def _send_admin_copy_email(
    *,
    job_id: str,
    name: str,
    submitter_email: str,
    cv_path: Path,
    starmap_path: Path,
    excel_path: Path,
) -> str | None:
    if not ADMIN_COPY_ENABLED or not ADMIN_COPY_EMAIL:
        return None
    body = (
        "Academic Star Map admin copy\n\n"
        f"Job ID: {job_id}\n"
        f"Researcher: {name}\n"
        f"Submitter email: {submitter_email or 'not provided'}\n"
        f"Download page: {_make_download_url(job_id)}\n\n"
        "Attached files: submitted CV, starmap HTML, and combined Excel workbook.\n"
    )
    return _send_email(
        ADMIN_COPY_EMAIL,
        subject=f"Academic Star Map admin copy: {name}",
        body=body,
        attachments=[cv_path, starmap_path, excel_path],
    )


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if str(path) and path.is_file():
            return path
    return None


def _find_first(run_dir: Path, pattern: str) -> Path | None:
    return next((path for path in sorted(run_dir.glob(pattern)) if path.is_file()), None)


def _build_combined_excel(run_dir: Path, output_path: Path) -> Path:
    israel_csv = _find_first(run_dir, "israel_top*.csv")
    world_csv = _find_first(run_dir, "world_top*.csv")
    if not israel_csv or not world_csv:
        missing = []
        if not israel_csv:
            missing.append("Israel CSV")
        if not world_csv:
            missing.append("World CSV")
        raise FileNotFoundError(f"Missing {', '.join(missing)} output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.read_csv(israel_csv).to_excel(writer, sheet_name="Israel", index=False)
        pd.read_csv(world_csv).to_excel(writer, sheet_name="World", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                header = str(column_cells[0].value or "")
                max_len = max(
                    [len(header)]
                    + [len(str(cell.value or "")) for cell in column_cells[1:]]
                )
                sheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_len + 2, 10),
                    42,
                )
    return output_path


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
            _delete_repo_file(record.get("starmap_repo_path"))
            _delete_repo_file(record.get("excel_repo_path"))
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
) -> None:
    try:
        _update_job(job_id, status="running", message="Parsing CV", started_at=_iso())
        cv_data = parse_cv(cv_path)

        admin_cv_path = None
        if ADMIN_COPY_ENABLED:
            admin_cv_path = _upload_file(cv_path, f"admin_copies/{job_id}/cv/{cv_path.name}")
            _update_job(job_id, admin_cv_path=admin_cv_path, admin_copy=True)

        if not ADMIN_COPY_ENABLED:
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
        starmap_src = _first_existing(
            [
                Path(result.get("star_map_path", "")),
                run_dir / "star_map.html",
            ]
        )
        if not starmap_src:
            raise FileNotFoundError("The star map HTML output was not created.")

        starmap_path = _result_starmap_path(job_id)
        starmap_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(starmap_src, starmap_path)

        excel_path = _build_combined_excel(run_dir, _result_excel_path(job_id))
        starmap_artifact = _upload_file(starmap_path, f"results/{job_id}/{starmap_path.name}")
        excel_artifact = _upload_file(excel_path, f"results/{job_id}/{excel_path.name}")

        warning = _send_ready_email(email, job_id)
        admin_warning = _send_admin_copy_email(
            job_id=job_id,
            name=name,
            submitter_email=email,
            cv_path=cv_path,
            starmap_path=starmap_path,
            excel_path=excel_path,
        )
        try:
            cv_path.unlink(missing_ok=True)
        except Exception:
            pass
        _update_job(
            job_id,
            status="completed",
            message="Ready to download",
            completed_at=_iso(),
            starmap_path=str(starmap_path),
            excel_path=str(excel_path),
            starmap_repo_path=starmap_artifact,
            excel_repo_path=excel_artifact,
            admin_cv_path=admin_cv_path,
            email_warning=warning,
            admin_email_warning=admin_warning,
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
        admin_copy=ADMIN_COPY_ENABLED,
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
    )
    return (
        f"Job `{job_id}` is queued. You can close the page; a notification email will be sent if email delivery is configured.",
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
        lines.append("The starmap and Excel file are ready.")
        if record.get("email_warning"):
            lines.append(f"Email note: {record['email_warning']}")
    if status == "expired":
        lines.append("These result files have expired and were deleted.")
    return "\n\n".join(lines)


def _mark_downloaded(job_id: str, record: dict[str, Any]) -> None:
    if not record.get("downloaded_at"):
        _update_job(job_id, downloaded_at=_iso())


def _prepare_named_result(
    job_id: str,
    *,
    local_key: str,
    repo_key: str,
    target_path: Path,
    label: str,
) -> Path | None:
    job_id = (job_id or "").strip()
    if not job_id:
        return None
    _cleanup_expired_jobs()
    record = _get_job(job_id)
    if not record:
        return None
    if record.get("status") != "completed":
        return None

    local_path = Path(record.get(local_key) or target_path)
    if local_path.suffix.lower() != target_path.suffix.lower():
        local_path = target_path
    if not local_path.exists():
        local_path = (
            _download_file_from_repo(record.get(repo_key, ""), target_path)
            or local_path
        )
    if not local_path.exists():
        raise gr.Error(f"The {label} file is missing. Please contact the site owner.")
    if local_path.suffix.lower() != target_path.suffix.lower():
        raise gr.Error(f"The {label} artifact has the wrong file type. Please rerun the job.")

    _mark_downloaded(job_id, record)
    return local_path


def download_starmap(job_id: str) -> str | None:
    path = _prepare_named_result(
        job_id,
        local_key="starmap_path",
        repo_key="starmap_repo_path",
        target_path=_result_starmap_path(job_id),
        label="starmap",
    )
    return str(path) if path else None


def download_excel(job_id: str) -> str | None:
    path = _prepare_named_result(
        job_id,
        local_key="excel_path",
        repo_key="excel_repo_path",
        target_path=_result_excel_path(job_id),
        label="Excel",
    )
    return str(path) if path else None


def _build_app() -> gr.Blocks:
    css = """
    body {
        background-color: #080a14;
        overflow-x: hidden;
    }
    #root,
    .app,
    main,
    .gradio-container {
        max-width: none !important;
        background: transparent !important;
        color: #f5f7fb;
    }
    .asm-space-bg {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(circle at 18% 28%, rgba(80, 210, 190, 0.18), transparent 13rem),
            radial-gradient(circle at 78% 78%, rgba(205, 85, 210, 0.13), transparent 15rem),
            radial-gradient(circle at 70% 18%, rgba(150, 230, 90, 0.11), transparent 14rem),
            radial-gradient(circle at 84% 46%, rgba(255, 142, 92, 0.11), transparent 12rem),
            #080a14;
    }
    .asm-space-bg::before,
    .asm-space-bg::after {
        content: "";
        position: absolute;
        inset: 0;
        background-repeat: no-repeat;
    }
    .asm-space-bg::before {
        opacity: 0.82;
        background-image:
            radial-gradient(circle at 3% 74%, rgba(255,255,255,0.58) 0 1px, transparent 2px),
            radial-gradient(circle at 7% 18%, rgba(255,255,255,0.34) 0 1px, transparent 2px),
            radial-gradient(circle at 11% 43%, rgba(140,217,255,0.42) 0 1.2px, transparent 2.4px),
            radial-gradient(circle at 14% 86%, rgba(255,255,255,0.26) 0 1px, transparent 2px),
            radial-gradient(circle at 19% 11%, rgba(255,255,255,0.42) 0 1px, transparent 2px),
            radial-gradient(circle at 23% 67%, rgba(255,255,255,0.36) 0 1.1px, transparent 2.2px),
            radial-gradient(circle at 27% 31%, rgba(255,255,255,0.30) 0 1px, transparent 2px),
            radial-gradient(circle at 31% 79%, rgba(140,217,255,0.44) 0 1.3px, transparent 2.6px),
            radial-gradient(circle at 34% 16%, rgba(255,255,255,0.22) 0 1px, transparent 2px),
            radial-gradient(circle at 39% 53%, rgba(255,255,255,0.50) 0 1.4px, transparent 2.8px),
            radial-gradient(circle at 44% 92%, rgba(255,255,255,0.34) 0 1px, transparent 2px),
            radial-gradient(circle at 49% 24%, rgba(255,217,0,0.30) 0 1.1px, transparent 2.3px),
            radial-gradient(circle at 54% 70%, rgba(255,255,255,0.42) 0 1px, transparent 2px),
            radial-gradient(circle at 58% 38%, rgba(255,255,255,0.28) 0 1px, transparent 2px),
            radial-gradient(circle at 62% 8%, rgba(140,217,255,0.36) 0 1.2px, transparent 2.4px),
            radial-gradient(circle at 68% 58%, rgba(255,255,255,0.52) 0 1.4px, transparent 2.8px),
            radial-gradient(circle at 73% 29%, rgba(255,255,255,0.32) 0 1px, transparent 2px),
            radial-gradient(circle at 77% 84%, rgba(255,255,255,0.42) 0 1.1px, transparent 2.2px),
            radial-gradient(circle at 82% 12%, rgba(255,255,255,0.48) 0 1px, transparent 2px),
            radial-gradient(circle at 88% 64%, rgba(140,217,255,0.38) 0 1.3px, transparent 2.6px),
            radial-gradient(circle at 94% 36%, rgba(255,255,255,0.30) 0 1px, transparent 2px),
            radial-gradient(circle at 97% 91%, rgba(255,255,255,0.36) 0 1.1px, transparent 2.2px);
    }
    .asm-space-bg::after {
        opacity: 0.72;
        background-image:
            radial-gradient(circle at 5% 32%, rgba(255,255,255,0.20) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 9% 58%, rgba(255,255,255,0.24) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 16% 22%, rgba(255,255,255,0.16) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 21% 74%, rgba(255,255,255,0.20) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 29% 7%, rgba(255,255,255,0.22) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 33% 61%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 41% 34%, rgba(255,255,255,0.26) 0 0.9px, transparent 1.9px),
            radial-gradient(circle at 47% 81%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 52% 13%, rgba(255,255,255,0.22) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 57% 49%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 64% 76%, rgba(255,255,255,0.28) 0 0.9px, transparent 1.9px),
            radial-gradient(circle at 69% 40%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 75% 6%, rgba(255,255,255,0.22) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 81% 53%, rgba(255,255,255,0.20) 0 0.8px, transparent 1.8px),
            radial-gradient(circle at 86% 25%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px),
            radial-gradient(circle at 91% 78%, rgba(255,255,255,0.26) 0 0.9px, transparent 1.9px),
            radial-gradient(circle at 96% 14%, rgba(255,255,255,0.18) 0 0.7px, transparent 1.7px);
    }
    .asm-comet {
        position: fixed;
        width: 120px;
        height: 2px;
        z-index: 1;
        pointer-events: none;
        transform-origin: center;
        background: linear-gradient(90deg, transparent, rgba(210, 235, 255, 0.36), rgba(255,255,255,0.70));
        filter: drop-shadow(0 0 5px rgba(140,217,255,0.24));
        opacity: 0.34;
    }
    .asm-comet.one {
        top: 17%;
        left: 9%;
        width: 150px;
        transform: rotate(21deg);
    }
    .asm-comet.two {
        top: 48%;
        right: 12%;
        width: 92px;
        transform: rotate(-18deg);
        opacity: 0.22;
    }
    .asm-comet.three {
        bottom: 14%;
        left: 24%;
        width: 72px;
        transform: rotate(12deg);
        opacity: 0.18;
    }
    .asm-shell,
    .asm-hero,
    .asm-panel {
        position: relative;
        z-index: 2;
    }
    .asm-shell {
        max-width: 1180px;
        margin: 0 auto;
        padding: 22px 18px 30px;
    }
    .asm-hero {
        background: rgba(8, 10, 20, 0.80);
        color: #f8fafc;
        border: 1px solid rgba(210, 225, 245, 0.16);
        border-radius: 8px;
        padding: 24px 26px;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
    }
    .asm-hero h1 {
        margin: 0 0 8px;
        font-size: 34px;
        line-height: 1.06;
        letter-spacing: 0;
        color: #ffffff;
    }
    .asm-hero p {
        margin: 0;
        max-width: 760px;
        color: #dce6ef;
        font-size: 16px;
        line-height: 1.5;
    }
    .asm-intro {
        margin: 18px 0;
        color: #c9d6e8;
        line-height: 1.55;
    }
    .asm-grid {
        gap: 16px;
        align-items: stretch;
    }
    .asm-panel {
        background: rgba(15, 22, 38, 0.84);
        border: 1px solid rgba(210, 225, 245, 0.14);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.34);
    }
    .asm-panel h2 {
        margin: 0 0 12px;
        font-size: 18px;
        line-height: 1.25;
        color: #d9e7f5;
        letter-spacing: 0;
    }
    .asm-status {
        min-height: 150px;
        padding: 14px;
        background: rgba(8, 10, 20, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        color: #eff6ff;
    }
    .asm-status p {
        margin-bottom: 8px;
    }
    .asm-downloads {
        gap: 10px;
    }
    .asm-downloads button {
        min-height: 44px;
        font-weight: 650;
    }
    .asm-panel button,
    .asm-panel button.primary {
        background: rgba(16, 22, 35, 0.94) !important;
        border: 1px solid rgba(220, 230, 245, 0.20) !important;
        color: #d8e0ee !important;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.04) !important;
    }
    .asm-panel button:hover,
    .asm-panel button.primary:hover {
        background: rgba(24, 31, 48, 0.98) !important;
        border-color: rgba(243, 217, 77, 0.32) !important;
        color: #f3f6fb !important;
    }
    .asm-downloads button:first-child {
        background: rgba(18, 28, 42, 0.94) !important;
    }
    .asm-panel label,
    .asm-panel span,
    .asm-panel .wrap,
    .asm-panel .prose,
    .asm-panel p {
        color: #e5eefc !important;
    }
    .asm-panel input,
    .asm-panel textarea,
    .asm-panel .file-preview {
        background: rgba(8, 10, 20, 0.86) !important;
        color: #f8fafc !important;
        border-color: rgba(140, 217, 255, 0.22) !important;
    }
    .asm-panel input::placeholder,
    .asm-panel textarea::placeholder {
        color: transparent !important;
    }
    .asm-footer {
        margin-top: 18px;
        color: #aab8cf;
        font-size: 14px;
        line-height: 1.5;
    }
    @media (max-width: 760px) {
        .asm-shell {
            padding: 12px;
        }
        .asm-hero {
            padding: 20px;
        }
        .asm-hero h1 {
            font-size: 28px;
        }
        .asm-panel {
            padding: 14px;
        }
    }
    """
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="yellow",
        neutral_hue="slate",
        radius_size="sm",
    )
    with gr.Blocks(title=CONTENT["brand"], css=css, theme=theme) as demo:
        gr.HTML(
            f"""
            <div class="asm-space-bg" aria-hidden="true"></div>
            <div class="asm-comet one" aria-hidden="true"></div>
            <div class="asm-comet two" aria-hidden="true"></div>
            <div class="asm-comet three" aria-hidden="true"></div>
            <main class="asm-shell">
              <section class="asm-hero">
                <h1>{CONTENT['brand']}</h1>
                <p>{CONTENT['tagline']}</p>
              </section>
            </main>
            """
        )
        with gr.Column(elem_classes=["asm-shell"]):
            gr.Markdown(CONTENT["intro_md"], elem_classes=["asm-intro"])

            with gr.Row(elem_classes=["asm-grid"]):
                with gr.Column(scale=1, elem_classes=["asm-panel"]):
                    gr.HTML("<h2>Submit CV</h2>")
                    name = gr.Textbox(label="Researcher full name")
                    institution = gr.Textbox(label="Institution hint (optional)")
                    orcid = gr.Textbox(label="ORCID (optional)")
                    submitter_email = gr.Textbox(label="Email for notification")
                    faculty_page = gr.Textbox(label="Faculty page URL (optional)")
                    cv_file = gr.File(
                        label="CV file (PDF or TXT, max 20 MB)",
                        file_types=[".pdf", ".txt"],
                        type="filepath",
                    )
                    submit = gr.Button("Create star map", variant="primary")
                with gr.Column(scale=1, elem_classes=["asm-panel"]):
                    gr.HTML("<h2>Results</h2>")
                    job_id = gr.Textbox(label="Job ID")
                    status = gr.Markdown("Submit a CV to start.", elem_classes=["asm-status"])
                    check = gr.Button("Check status")
                    with gr.Row(elem_classes=["asm-downloads"]):
                        gr.DownloadButton(
                            "Download starmap",
                            value=download_starmap,
                            inputs=[job_id],
                            variant="secondary",
                        )
                        gr.DownloadButton(
                            "Download Excel",
                            value=download_excel,
                            inputs=[job_id],
                            variant="secondary",
                        )

            submit.click(
                start_job,
                inputs=[
                    name,
                    institution,
                    orcid,
                    submitter_email,
                    faculty_page,
                    cv_file,
                ],
                outputs=[status, job_id],
            )
            check.click(check_status, inputs=[job_id], outputs=[status])

            gr.Markdown(CONTENT["privacy_note_md"], elem_classes=["asm-footer"])

    return demo


if __name__ == "__main__":
    app = _build_app()
    app.queue(default_concurrency_limit=MAX_WORKERS).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
