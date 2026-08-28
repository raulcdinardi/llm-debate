#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo:" >&2
  echo "  sudo bash scripts/install_docker_wsl.sh" >&2
  exit 2
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Cannot find /etc/os-release; this script expects Ubuntu on WSL." >&2
  exit 2
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script expects Ubuntu. Found ID=${ID:-unknown}." >&2
  exit 2
fi

TARGET_USER="${SUDO_USER:-}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
  TARGET_USER="$(logname 2>/dev/null || true)"
fi
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
  echo "Could not determine the non-root user to add to the docker group." >&2
  exit 2
fi

apt-get update
apt-get install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
fi
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

usermod -aG docker "${TARGET_USER}"

if command -v systemctl >/dev/null 2>&1 && [[ "$(ps -p 1 -o comm=)" == "systemd" ]]; then
  systemctl enable --now docker
else
  service docker start
fi

docker run --rm hello-world

cat <<EOF

Docker installed and the daemon smoke test passed.

User '${TARGET_USER}' was added to the docker group. Restart WSL before running
docker without sudo:

  powershell.exe -NoProfile -Command "wsl --shutdown"

Then reopen WSL and verify:

  docker run --rm hello-world
EOF
