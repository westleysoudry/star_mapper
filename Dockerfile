FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src
COPY content ./content
COPY app.py ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "app.py"]
