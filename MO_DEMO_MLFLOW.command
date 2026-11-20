#!/bin/zsh
# Mở demo MLflow trên chính máy Mac này.
set -e

cd "${0:A:h}"

if [[ -x ../.runtime/venv/bin/python ]]; then
  PYTHON=../.runtime/venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

URL=http://127.0.0.1:5000

open_demo() {
  print "MLflow đang mở tại $URL"
  if [[ "${MLFLOW_DEMO_NO_OPEN:-}" != 1 ]]; then
    open "$URL"
  fi
  print "Trong giao diện: Model training → FlightDelay-Tournament-Reported → Runs"
}

if curl -fsS --max-time 2 -o /dev/null "$URL"; then
  open_demo
  exit 0
fi

if ! grep -l 'FlightDelay-Tournament-Reported' mlruns/*/meta.yaml >/dev/null 2>&1; then
  print "Đang chuẩn bị 21 run demo từ kết quả nhóm đã ghi trong repo..."
  "$PYTHON" scripts/log_tournament_mlflow.py
fi

print "Đang bật MLflow UI. Giữ cửa sổ Terminal này mở trong lúc thuyết trình."
"$PYTHON" -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

for attempt in {1..30}; do
  if curl -fsS --max-time 2 -o /dev/null "$URL"; then
    open_demo
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 1
done

print "MLflow chưa mở được. Hãy chụp màn hình lỗi trong cửa sổ này để mình kiểm tra."
exit 1
