#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/data/raw"

mkdir -p "$RAW_DIR"

echo "========================================"
echo " Descargando datasets para FSCIL-Lab"
echo "========================================"

# ──────────────────────────────────────────────
# 1. CIFAR-100
# ──────────────────────────────────────────────
echo ""
echo "[1/3] CIFAR-100..."
CIFAR_DIR="$RAW_DIR/cifar100"
if [ -f "$CIFAR_DIR/train" ] && [ -f "$CIFAR_DIR/test" ] && [ -f "$CIFAR_DIR/meta" ]; then
    echo "  Ya existe en $CIFAR_DIR — saltando."
else
    mkdir -p "$CIFAR_DIR"
    URL_CIFAR="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    # Intentar con aria2c (multi-thread) o axel, si no con wget
    if command -v aria2c &>/dev/null; then
        echo "  Descargando con aria2c (multi-thread)..."
        aria2c -x 4 -s 4 --continue -d /tmp -o cifar-100-python.tar.gz "$URL_CIFAR"
    elif command -v axel &>/dev/null; then
        echo "  Descargando con axel (multi-thread)..."
        axel -n 4 -o /tmp/cifar-100-python.tar.gz "$URL_CIFAR"
    else
        echo "  Descargando con wget (instala 'aria2c' para multi-thread)..."
        wget --continue --show-progress "$URL_CIFAR" -O /tmp/cifar-100-python.tar.gz
    fi
    tar -xzf /tmp/cifar-100-python.tar.gz -C /tmp/
    mv /tmp/cifar-100-python/train /tmp/cifar-100-python/test /tmp/cifar-100-python/meta "$CIFAR_DIR/"
    rm -rf /tmp/cifar-100-python /tmp/cifar-100-python.tar.gz
    echo "  ✓ CIFAR-100 en $CIFAR_DIR"
fi

# ──────────────────────────────────────────────
# 2. PlantVillage (via Hugging Face — más rápido)
# ──────────────────────────────────────────────
echo ""
echo "[2/3] PlantVillage..."
PV_DIR="$RAW_DIR/plantvillage"
PV_COLOR="$PV_DIR/color"
if [ -d "$PV_COLOR" ] && ls "$PV_COLOR"/*/ >/dev/null 2>&1; then
    echo "  Ya existe en $PV_COLOR — saltando."
else
    mkdir -p "$PV_DIR"
    echo "  Instalando huggingface-hub..."
    pip install -q huggingface-hub
    echo "  Descargando PlantVillage desde Hugging Face..."
    python3 -c "
from huggingface_hub import snapshot_download
import os, shutil

destino = os.path.join('$PV_DIR', 'hf_temp')
snapshot_download(repo_id='mohanty/PlantVillage', repo_type='dataset', local_dir=destino)

# Mover raw/color/ a la estructura esperada
origen_color = os.path.join(destino, 'raw', 'color')
destino_color = os.path.join('$PV_DIR', 'color')
if os.path.exists(origen_color):
    shutil.move(origen_color, destino_color)
    shutil.rmtree(destino, ignore_errors=True)
    print('  ✓ PlantVillage en $PV_COLOR')
else:
    # Fallback: mover todo lo descargado
    shutil.move(destino, destino_color)
    print('  ✓ PlantVillage descargado (estructura alternativa)')
"
fi

# ──────────────────────────────────────────────
# 3. PlantDoc
# ──────────────────────────────────────────────
echo ""
echo "[3/3] PlantDoc..."
PDOC_DIR="$RAW_DIR/plantdoc"
if [ -d "$PDOC_DIR" ] && ls "$PDOC_DIR"/*/ >/dev/null 2>&1; then
    echo "  Ya existe en $PDOC_DIR — saltando."
else
    mkdir -p "$PDOC_DIR"
    echo "  Descargando PlantDoc..."
    wget -q --show-progress https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip -O /tmp/plantdoc.zip
    unzip -q /tmp/plantdoc.zip -d /tmp/plantdoc_extracted
    mv /tmp/plantdoc_extracted/PlantDoc-Dataset-master/* "$PDOC_DIR/"
    rm -rf /tmp/plantdoc.zip /tmp/plantdoc_extracted
    echo "  ✓ PlantDoc en $PDOC_DIR"
fi

echo ""
echo "========================================"
echo " Descarga completada."
echo " Ejecuta: streamlit run fscil_lab/app.py"
echo "========================================"
