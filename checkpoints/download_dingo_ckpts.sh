#!/bin/bash

URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
FILE="groundingdino_swint_ogc.pth"

if [ -f "$FILE" ]; then
    echo "$FILE already exists. Skipping download."
else
    echo "Downloading Grounding DINO weights ($FILE)..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$URL"
    else
        curl -L -O "$URL"
    fi
fi