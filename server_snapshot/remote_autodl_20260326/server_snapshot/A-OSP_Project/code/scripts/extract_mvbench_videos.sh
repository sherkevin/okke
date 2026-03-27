#!/bin/bash
# Extract MVBench video zips and locate action_sequence videos
# Run after sta.zip download completes

PROJECT_ROOT="/root/autodl-tmp/A-OSP_Project"
VIDEO_DIR="$PROJECT_ROOT/data/mvbench/video"

echo "=== MVBench Video Extraction ==="
echo "Video dir: $VIDEO_DIR"

# Extract sta.zip (main source for action_sequence)
if [ -f "$VIDEO_DIR/sta.zip" ]; then
    echo "[1/2] Extracting sta.zip..."
    mkdir -p "$VIDEO_DIR/extracted"
    unzip -q "$VIDEO_DIR/sta.zip" -d "$VIDEO_DIR/extracted/" 2>/dev/null || true
    echo "  Extracted. Contents:"
    ls "$VIDEO_DIR/extracted/" | head -10
    # Count mp4s
    n_mp4=$(find "$VIDEO_DIR/extracted" -name "*.mp4" | wc -l)
    echo "  Found $n_mp4 .mp4 files"
else
    echo "[1/2] sta.zip not found, skipping."
fi

# Check for action_sequence videos specifically
echo ""
echo "[2/2] Checking for action_sequence videos..."
ACTION_SEQ_JSON="$PROJECT_ROOT/data/mvbench/json/action_sequence.json"
if [ -f "$ACTION_SEQ_JSON" ]; then
    # Get first 10 video filenames
    python3 -c "
import json
with open('$ACTION_SEQ_JSON') as f:
    data = json.load(f)
filenames = [d['video'] for d in data[:10]]
print('First 10 target videos:')
for fn in filenames:
    print(' ', fn)
"
fi

# Search for them
echo ""
echo "Searching extracted directories..."
for search_dir in "$VIDEO_DIR/extracted" "$VIDEO_DIR"; do
    if [ -d "$search_dir" ]; then
        n=$(find "$search_dir" -name "ZS9XR.mp4" 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then
            echo "  FOUND ZS9XR.mp4 in: $(find "$search_dir" -name "ZS9XR.mp4" | head -1)"
        fi
    fi
done

echo ""
echo "=== Done ==="
