#!/usr/bin/env bash
# ============================================================
# Task 3: CHAIR Evaluation Tooling Setup
# ============================================================
# Downloads the standalone CHAIR metric files and the COCO
# instances_val2014.json ground-truth annotations.
#
# NOTE: GitHub clone may fail on restricted networks (e.g. AutoDL).
#       This script falls back to downloading individual files
#       from raw.githubusercontent.com when clone fails.
#
# Usage:
#   bash code/data/setup_chair.sh
#
# Result layout:
#   data/chair/
#     CHAIR-metric-standalone/
#       chair.py                  ← core evaluation script
#       chair.pkl                 ← pre-built cache (optional)
#       example_inputs.jsonl      ← example format
#     coco_annotations/
#       instances_val2014.json    ← COCO GT (~80 MB)
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHAIR_DIR="${PROJECT_ROOT}/data/chair"
REPO_DIR="${CHAIR_DIR}/CHAIR-metric-standalone"
ANNO_DIR="${CHAIR_DIR}/coco_annotations"

CHAIR_REPO="https://github.com/Maxlinn/CHAIR-metric-standalone.git"
CHAIR_RAW_BASE="https://raw.githubusercontent.com/Maxlinn/CHAIR-metric-standalone/main"
COCO_ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

CHAIR_FILES=(
    "chair.py"
    "example_inputs.jsonl"
    "README.md"
)

echo "============================================================"
echo "A-OSP Data Infrastructure — Task 3: CHAIR Setup"
echo "============================================================"
echo "Project root : ${PROJECT_ROOT}"
echo "CHAIR dir    : ${CHAIR_DIR}"
echo ""

mkdir -p "${CHAIR_DIR}"
mkdir -p "${REPO_DIR}"

# ── Step 1: Get CHAIR code ────────────────────────────────────
echo "──────────────────────────────────────────────────────"
echo "Step 1: Obtain CHAIR-metric-standalone"
echo "──────────────────────────────────────────────────────"

if [ -f "${REPO_DIR}/chair.py" ]; then
    echo "[CHAIR] chair.py already exists. Skipping download."
else
    CLONE_OK=false
    echo "[CHAIR] Attempting git clone (timeout 15s) ..."
    if timeout 15 git clone --depth 1 "${CHAIR_REPO}" "${REPO_DIR}" 2>/dev/null; then
        CLONE_OK=true
        echo "[CHAIR] Clone succeeded."
    else
        echo "[CHAIR] Clone failed (network restriction likely). Falling back to direct download..."
        rm -rf "${REPO_DIR}"
        mkdir -p "${REPO_DIR}"

        for fname in "${CHAIR_FILES[@]}"; do
            echo "[CHAIR] Downloading ${fname} ..."
            if curl -fsSL --connect-timeout 30 --max-time 120 \
                   -o "${REPO_DIR}/${fname}" \
                   "${CHAIR_RAW_BASE}/${fname}"; then
                echo "[CHAIR]   ✓ ${fname} ($(wc -c < "${REPO_DIR}/${fname}") bytes)"
            else
                echo "[CHAIR]   ✗ Failed to download ${fname}"
            fi
        done

        # chair.pkl is a binary blob (~2MB), needs special handling via GitHub API
        echo "[CHAIR] Downloading chair.pkl (pre-built cache, ~2MB) ..."
        CHAIR_PKL_SHA="8a21f58aebf532e8307c7f18702b5abaa3a2347c"
        if curl -fsSL --connect-timeout 30 --max-time 120 \
               -H "Accept: application/vnd.github.raw" \
               -o "${REPO_DIR}/chair.pkl" \
               "https://api.github.com/repos/Maxlinn/CHAIR-metric-standalone/git/blobs/${CHAIR_PKL_SHA}"; then
            echo "[CHAIR]   ✓ chair.pkl ($(wc -c < "${REPO_DIR}/chair.pkl") bytes)"
        else
            echo "[CHAIR]   ✗ chair.pkl download failed (can rebuild with: python chair.py --cache)"
        fi
    fi
fi

echo ""
echo "[CHAIR] Repo contents:"
ls -la "${REPO_DIR}/" 2>/dev/null || echo "(empty)"

# ── Step 2: Download COCO annotations ─────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 2: Download COCO instances_val2014.json"
echo "──────────────────────────────────────────────────────"

mkdir -p "${ANNO_DIR}"
TARGET_FILE="${ANNO_DIR}/instances_val2014.json"

if [ -f "${TARGET_FILE}" ]; then
    echo "[COCO] instances_val2014.json already exists ($(du -h "${TARGET_FILE}" | cut -f1))."
    echo "[COCO] Skipping download."
else
    ZIP_PATH="${CHAIR_DIR}/annotations_trainval2014.zip"
    echo "[COCO] Downloading COCO annotations zip ..."
    echo "[COCO] URL: ${COCO_ANNO_URL}"
    echo "[COCO] This is ~252 MB, please wait ..."

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${ZIP_PATH}" "${COCO_ANNO_URL}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${ZIP_PATH}" "${COCO_ANNO_URL}"
    else
        echo "[COCO] ERROR: Neither wget nor curl available."
        exit 1
    fi

    echo "[COCO] Extracting instances_val2014.json ..."
    unzip -o -j "${ZIP_PATH}" "annotations/instances_val2014.json" -d "${ANNO_DIR}"

    if [ -f "${TARGET_FILE}" ]; then
        echo "[COCO] Extracted: ${TARGET_FILE} ($(du -h "${TARGET_FILE}" | cut -f1))"
        echo "[COCO] Cleaning up zip file ..."
        rm -f "${ZIP_PATH}"
    else
        echo "[COCO] ERROR: Extraction failed. Keeping zip for manual inspection."
        echo "[COCO] Zip location: ${ZIP_PATH}"
        exit 1
    fi
fi

# ── Step 3: Symlink annotations into CHAIR dir ───────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 3: Link annotations into CHAIR dir"
echo "──────────────────────────────────────────────────────"

CHAIR_COCO_DIR="${REPO_DIR}/coco_annotations"
mkdir -p "${CHAIR_COCO_DIR}"

if [ ! -L "${CHAIR_COCO_DIR}/instances_val2014.json" ] && [ ! -f "${CHAIR_COCO_DIR}/instances_val2014.json" ]; then
    ln -sf "${TARGET_FILE}" "${CHAIR_COCO_DIR}/instances_val2014.json"
    echo "[CHAIR] Symlinked instances_val2014.json into repo dir."
else
    echo "[CHAIR] Annotation link already exists."
fi

# ── Step 4: Verify setup ─────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 4: Verification"
echo "──────────────────────────────────────────────────────"

PASS=true

if [ -f "${REPO_DIR}/chair.py" ]; then
    SIZE=$(wc -c < "${REPO_DIR}/chair.py")
    echo "[CHECK] chair.py                ✓  (${SIZE} bytes)"
else
    echo "[CHECK] chair.py                ✗  MISSING"
    PASS=false
fi

if [ -f "${TARGET_FILE}" ]; then
    SIZE=$(du -h "${TARGET_FILE}" | cut -f1)
    echo "[CHECK] instances_val2014.json  ✓  (${SIZE})"
else
    echo "[CHECK] instances_val2014.json  ✗  MISSING (download separately)"
fi

if [ -f "${REPO_DIR}/chair.pkl" ]; then
    SIZE=$(du -h "${REPO_DIR}/chair.pkl" | cut -f1)
    echo "[CHECK] chair.pkl (cache)       ✓  (${SIZE})"
else
    echo "[CHECK] chair.pkl (cache)       -  Not present (rebuild with: python chair.py --cache)"
fi

SYMLINK="${CHAIR_COCO_DIR}/instances_val2014.json"
if [ -f "${SYMLINK}" ] || [ -L "${SYMLINK}" ]; then
    echo "[CHECK] annotation symlink      ✓"
else
    echo "[CHECK] annotation symlink      ✗  MISSING"
fi

echo ""
if [ "$PASS" = true ]; then
    echo "============================================================"
    echo "CHAIR SETUP COMPLETE (core files) ✓"
    echo "============================================================"
    echo ""
    echo "To build/rebuild the CHAIR cache (one-time):"
    echo "  cd ${REPO_DIR}"
    echo "  python chair.py --cache"
    echo ""
    echo "To evaluate captions:"
    echo "  python chair.py --cap_file <your_captions.json>"
    echo ""
else
    echo "============================================================"
    echo "CHAIR SETUP INCOMPLETE — see errors above"
    echo "============================================================"
    exit 1
fi
