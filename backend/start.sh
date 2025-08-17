#!/usr/bin/env bash

# ====== SIMPLE LICENSE GATE ======
SERVER_URL="https://license-checker.serpcompany.workers.dev"
LICENSE_FIELD="LICENSE_KEY_AI_VOICE_CLONER"
LICENSE_OK_FILE="${LICENSE_OK_FILE:-/tmp/.license_ok}"
USE_LICENSE_CHECK="${USE_LICENSE_CHECK:-true}"

if [[ "${USE_LICENSE_CHECK,,}" == "true" || "${USE_LICENSE_CHECK}" == "1" || "${USE_LICENSE_CHECK,,}" == "yes" ]]; then
  if [[ -f "$LICENSE_OK_FILE" ]]; then
    echo "✓ License previously validated ($(date -r "$LICENSE_OK_FILE" 2>/dev/null || echo "unknown time")) — starting app."
  else
    if [[ -z "${EMAIL:-}" ]]; then
      echo "ERROR: EMAIL env var is required=" >&2
      exit 1
    fi
    if [[ -z "${LICENSE_KEY:-}" ]]; then
      echo "ERROR: LICENSE_KEY env var is required" >&2
      exit 1
    fi

    payload=$(cat <<EOF
{"license_key":"${LICENSE_KEY}","email":"${EMAIL}","license_field":"${LICENSE_FIELD}"}
EOF
)

    echo "→ Verifying license for ${EMAIL} ..."
    if curl --version 2>/dev/null | grep -q 'fail-with-body'; then
      resp="$(curl -sS --fail-with-body -X POST "$SERVER_URL" \
        -H 'content-type: application/json' --data "$payload")" || {
          code=$?
          echo "✗ License endpoint error (curl exit $code)"; [[ -n "$resp" ]] && echo "$resp"
          exit 1
        }
    else
      resp="$(curl -sS -X POST "$SERVER_URL" \
        -H 'content-type: application/json' --data "$payload")" || {
          code=$?
          echo "✗ License endpoint error (curl exit $code)"; exit 1
        }
    fi

    if echo "$resp" | grep -q '"valid"[[:space:]]*:[[:space:]]*true'; then
      echo "✓ License validated."
      printf '%s\n' "$resp" > "$LICENSE_OK_FILE" 2>/dev/null || touch "$LICENSE_OK_FILE"
    else
      echo "✗ License check failed. Server said:" >&2
      echo "$resp" >&2
      exit 1
    fi
  fi
else
  echo "USE_LICENSE_CHECK=false — skipping license check."
fi

TCMALLOC="$(ldconfig -p 2>/dev/null | grep -Po "libtcmalloc.so.\d" | head -n 1)"
[[ -n "$TCMALLOC" ]] && export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
export HF_HOME="/"

export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

echo "AI Inference APP - Starting backend"
echo "API will be available at http://${API_HOST}:${API_PORT}"

if [ -n "$API_KEY" ]; then
    echo "🔒 API Key authentication enabled"
else
    echo "🔓 API Key authentication disabled (set API_KEY env var to enable)"
fi

cd /src
exec uvicorn api_handler:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio
