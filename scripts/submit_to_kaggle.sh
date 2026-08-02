#!/bin/bash
# Submits the built tarball to Kaggle
echo "Submitting to Kaggle..."
kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m "RL Agent (Phase 10 - Ep 20k)"
echo "Submission complete. Check leaderboard for results."
