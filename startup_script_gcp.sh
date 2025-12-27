#!/bin/bash
set -euxo pipefail

APP_USER="rinat227722"
APP_HOME="/home/${APP_USER}"
VENV_DIR="${APP_HOME}/venv"

# Ensure user exists
if ! id "${APP_USER}" &>/dev/null; then
  useradd -m -s /bin/bash "${APP_USER}"
fi

apt-get update
apt-get install -y python3-venv python3-pip

# Create venv as the normal user
sudo -u "${APP_USER}" bash -lc "
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
pip install --upgrade pip
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
pip install \
  Flask==2.0.2 \
  Werkzeug==2.3.8 \
  flask-restful==0.3.9 \
  nltk==3.6.3 \
  pandas \
  google-cloud-storage==3.7.0 \
  sentence-transformers==3.2.1 \
  'numpy>=1.23.2,<3' \
  pyarrow==22.0.0
"
