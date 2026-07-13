#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/data/raw"

echo "========================================"
echo " Organizando datasets para FSCIL-Lab"
echo "========================================"

# ──────────────────────────────────────────────
# PlantVillage
# ──────────────────────────────────────────────
echo ""
echo "[1/2] PlantVillage..."
PV_DIR="$RAW_DIR/plantvillage"
PV_COLOR="$PV_DIR/color"

# Quitar el placeholder anterior si existe
rm -rf "$PV_COLOR"

# Buscar donde están las imágenes reales
if [ -d "$PV_DIR/plantvillage dataset/color" ] && ls "$PV_DIR/plantvillage dataset/color"/*/ >/dev/null 2>&1; then
    echo "  Moviendo 'plantvillage dataset/color' -> 'color'..."
    mkdir -p "$PV_COLOR"
    mv "$PV_DIR/plantvillage dataset/color"/* "$PV_COLOR/"
    rm -rf "$PV_DIR/plantvillage dataset"
    echo "  ✓ Estructura PlantVillage corregida"
elif [ -f "$PV_DIR/color/data.zip" ]; then
    echo "  Extrayendo data.zip (2.1 GB, puede tomar un momento)..."
    unzip -q "$PV_DIR/color/data.zip" -d /tmp/pv_extracted
    mv /tmp/pv_extracted/* "$PV_COLOR/"
    rm -rf /tmp/pv_extracted "$PV_DIR/color/data.zip"
    echo "  ✓ data.zip extraído en color/"
else
    echo "  ! No se encontraron imágenes de PlantVillage."
    echo "    Vuelve a ejecutar: ./fscil_lab/download_datasets.sh"
fi

echo ""
echo "  Clases disponibles en color/:"
ls -d "$PV_COLOR"/*/ 2>/dev/null | sed 's|.*/||' | head -20
echo "  Total: $(ls -d "$PV_COLOR"/*/ 2>/dev/null | wc -l) clases"

# ──────────────────────────────────────────────
# PlantDoc
# ──────────────────────────────────────────────
echo ""
echo "[2/2] PlantDoc..."
PDOC_DIR="$RAW_DIR/plantdoc"

if [ -d "$PDOC_DIR/train" ] && ls "$PDOC_DIR/train"/*/ >/dev/null 2>&1; then
    echo "  Fusionando train/ y test/ en la raíz..."

    # Recolectar nombres de clases únicos de train y test
    CLASES=$( (ls "$PDOC_DIR/train/"; ls "$PDOC_DIR/test/") | sort -u)

    for clase in $CLASES; do
        mkdir -p "$PDOC_DIR/$clase"
        # Mover imágenes de train/
        if [ -d "$PDOC_DIR/train/$clase" ]; then
            mv "$PDOC_DIR/train/$clase"/* "$PDOC_DIR/$clase/" 2>/dev/null || true
        fi
        # Mover imágenes de test/
        if [ -d "$PDOC_DIR/test/$clase" ]; then
            mv "$PDOC_DIR/test/$clase"/* "$PDOC_DIR/$clase/" 2>/dev/null || true
        fi
    done

    rm -rf "$PDOC_DIR/train" "$PDOC_DIR/test" "$PDOC_DIR/train.csv" "$PDOC_DIR/test.csv" 2>/dev/null || true
    rm -f "$PDOC_DIR/LICENSE.txt" "$PDOC_DIR/README.md" "$PDOC_DIR/PlantDoc_Examples.png" 2>/dev/null || true

    echo "  ✓ PlantDoc reorganizado"
else
    echo "  ! No se encontraron subdirectorios train/test en PlantDoc."
    ls "$PDOC_DIR" 2>/dev/null
fi

echo ""
echo "  Clases disponibles en plantdoc/:"
ls -d "$PDOC_DIR"/*/ 2>/dev/null | sed 's|.*/||' | head -20
echo "  Total: $(ls -d "$PDOC_DIR"/*/ 2>/dev/null | wc -l) clases"

echo ""
echo "========================================"
echo " Organización completada."
echo " Ejecuta: streamlit run fscil_lab/app.py"
echo "========================================"
