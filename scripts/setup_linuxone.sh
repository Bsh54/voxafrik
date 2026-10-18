#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
mkdir -p data/afrihate

echo "Dependencies installed."
echo "Next steps:"
echo "  1. Download the datasets:   python scripts/download_data.py"
echo "  2. Build the augmentation:  python backend/augment.py"
echo "  3. Start the API:           python backend/api.py"
