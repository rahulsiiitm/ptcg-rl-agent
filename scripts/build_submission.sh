#!/bin/bash
# Builds the submission.tar.gz for Kaggle

echo "Building submission..."
tar -czvf submission.tar.gz main.py deck.csv src/ models/ppo_phase3.pth data/ LICENSE
echo "Done. submission.tar.gz created."
