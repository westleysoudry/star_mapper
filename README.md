---
title: Academic Star Map
emoji: ⭐
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Academic Star Map

Academic Star Map builds an interactive researcher starmap from a CV and public scholarly metadata.

The editable website text lives in:

- `content/site.yaml`
- `content/emails/result_ready.md`

The main web app is `app.py`. The existing mapper package remains under `src/researcher_mapper`.

## Deployment

This repository is set up for a Hugging Face Spaces Docker deployment from GitHub Actions.

Default Space repo in `.github/workflows/deploy-space.yml`:

```text
westleysoudry/academic-star-map
```

If your Hugging Face username or Space name is different, set the GitHub Actions repository variable `HF_SPACE_REPO` to the real Space id, for example `your-hf-username/academic-star-map`. If no variable is set, the workflow uses the default above.

Required GitHub secret:

```text
HF_TOKEN
```

`HF_TOKEN` must be a Hugging Face token from an account that can create and write to the Space namespace in `HF_SPACE_REPO`. Your GitHub username and Hugging Face username do not have to match.

Required Hugging Face Space secrets / variables:

```text
HF_RESULTS_REPO=westleysoudry/academic-star-map-results
HF_DATA_TOKEN=<token with write access to the private results dataset>
OPENALEX_API_KEY=<free OpenAlex API key>
OPENALEX_EMAIL=WestleySoudry@gmail.com
CROSSREF_MAILTO=WestleySoudry@gmail.com
MAIL_FROM=WestleySoudry@gmail.com
RESEND_API_KEY=<Resend email API key>
# or BREVO_API_KEY=<Brevo transactional email API key>
PUBLIC_BASE_URL=<your Space URL>
ADMIN_COPY_ENABLED=true
ADMIN_COPY_EMAIL=westleysoudry@gmail.com
```

Optional:

```text
TEST_ACCESS_CODE=<shared private beta code>
TEST_ALLOWED_EMAILS=person1@example.com,person2@example.com
SEMANTIC_SCHOLAR_API_KEY=<Semantic Scholar API key>
EMAIL_BACKEND=brevo
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=WestleySoudry@gmail.com
SMTP_PASSWORD=<Google app password>
MAIL_FROM_NAME=Academic Star Map
MAX_WORKERS=1
MAX_CV_BYTES=20971520
```

Email delivery uses an HTTPS email API by default because some hosting platforms block outbound SMTP ports. Add `RESEND_API_KEY` or `BREVO_API_KEY`; SMTP is only used when `EMAIL_BACKEND=smtp`. For Brevo, `MAIL_FROM` should be a sender address verified in Brevo. If `MAIL_FROM` is not set, the app falls back to `ADMIN_COPY_EMAIL`.

For a private testing phase, set `TEST_ACCESS_CODE` to require a shared code before submission or status/download checks. Set `TEST_ALLOWED_EMAILS` to a comma-separated allowlist. If both are set, both checks must pass.

## Privacy Defaults

- Uploaded CVs are deleted immediately after parsing.
- With `ADMIN_COPY_ENABLED=true`, uploaded CVs and result files are sent to `ADMIN_COPY_EMAIL` after each completed job.
- Result files are deleted 1 day after download.
- Result files are deleted after 7 days if never downloaded.

## Local Run

```bash
pip install -e .
python app.py
```

Open the printed local URL.
