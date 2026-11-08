#!/bin/sh
# Single image, three roles: api | ui | demo (or any command).
set -e
DATA="${FORENSICS_DATA_DIR:-/app/data}"
mkdir -p "$DATA"

case "$1" in
  api)
    if [ "${FORENSICS_SEED_DEMO:-1}" = "1" ] && [ ! -f "$DATA/forensics.db" ]; then
      echo "[entrypoint] empty data volume -> generating demo scenario (50 docs x 3 pipeline versions)"
      python -m forensics.cli demo
    fi
    exec uvicorn forensics.api:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  ui)
    exec streamlit run ui/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" \
      --server.headless true --browser.gatherUsageStats false
    ;;
  demo)
    exec python -m forensics.cli demo
    ;;
  *)
    exec "$@"
    ;;
esac
