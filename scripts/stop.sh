#!/usr/bin/env bash
# scripts/stop.sh
# ============================================================
# One-command teardown for the Disease Prediction MLOps stack.
# Run from repository root:
#   ./scripts/stop.sh
#
# Flags:
#   -v, --volumes   Remove named Docker volumes (database data, etc.)
#   --rmi           Remove images built by Compose
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

REMOVE_VOLUMES=false
REMOVE_IMAGES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        --rmi)
            REMOVE_IMAGES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/stop.sh [-v|--volumes] [--rmi]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " Stopping Disease Prediction MLOps Platform"
echo "============================================================"

COMPOSE_ARGS=("down")

if [ "$REMOVE_VOLUMES" = true ]; then
    echo "[!] WARNING: Removing Docker volumes. All PostgreSQL/Grafana persistent data will be deleted."
    COMPOSE_ARGS+=("-v")
fi

if [ "$REMOVE_IMAGES" = true ]; then
    echo "[*] Removing locally built images..."
    COMPOSE_ARGS+=("--rmi" "local")
fi

docker compose "${COMPOSE_ARGS[@]}"

echo ""
echo "[+] All containers stopped successfully."
echo "============================================================"
