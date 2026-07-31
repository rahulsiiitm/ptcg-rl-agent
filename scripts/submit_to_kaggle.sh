#!/bin/bash
# Submits the built tarball to Kaggle
echo "Submitting to Kaggle..."
kaggle competitions submit -c pokemon-tcg-ai-battle-challenge-strategy -f submission.tar.gz -m "Rule-based Agent (Phase 1)"
echo "Submission complete. Check leaderboard for results."
