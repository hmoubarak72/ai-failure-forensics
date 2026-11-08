# ---- build stage: resolve wheels once --------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- runtime stage ------------------------------------------------------------
FROM python:3.11-slim
LABEL org.opencontainers.image.title="ai-failure-forensics" \
      org.opencontainers.image.description="Failure forensics for multi-step AI pipelines" \
      org.opencontainers.image.source="https://github.com/hmoubarak72/ai-failure-forensics"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORENSICS_DATA_DIR=/app/data \
    LLM_PROVIDER=auto \
    JUDGE_PROVIDER=auto \
    PIPELINE_VERSION=v1

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels \
 && useradd --create-home --uid 1000 forensics

COPY forensics ./forensics
COPY ui ./ui
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN mkdir -p /app/data && chown -R forensics:forensics /app

USER forensics
VOLUME ["/app/data"]
EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); u='http://127.0.0.1:'+p+('/_stcore/health' if p=='8501' else '/health'); urllib.request.urlopen(u, timeout=4)" || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["api"]
