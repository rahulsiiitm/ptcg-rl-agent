#!/bin/bash
# Builds the submission.tar.gz for Kaggle

echo "Building submission..."
tar -czvf submission.tar.gz main.py deck.csv src/ models/ppo_phase3.pth data/EN_Card_Data.csv data/card_lookup.py data/__init__.py
echo "Done. submission.tar.gz created."
