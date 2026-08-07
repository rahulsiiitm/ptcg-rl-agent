#!/bin/bash
set -euo pipefail

# Build a Kaggle-safe, versioned archive without local replay/PDF/cache files.
# Usage: bash scripts/build_submission.sh v22
version="${1:?usage: $0 VERSION (for example: v22)}"

test -d top20_decks
mkdir -p submissions

echo "Building ${version}..."
tar \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf submission.tar.gz \
    main.py \
    deck.csv \
    lucario_w.json \
    src/__init__.py \
    src/agent/__init__.py \
    src/agent/rule_based_lucario.py \
    cg/ \
    top20_decks/ \
    data/EN_Card_Data.csv \
    data/card_lookup.py \
    data/__init__.py

cp submission.tar.gz "${version}.tar.gz"
cp submission.tar.gz "submissions/${version}.tar.gz"

echo "Created submission.tar.gz, ${version}.tar.gz, and submissions/${version}.tar.gz"
echo "Size: $(du -sh submission.tar.gz | cut -f1)"
