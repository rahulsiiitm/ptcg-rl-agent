#!/bin/bash
# Builds the submission.tar.gz for Kaggle

echo "Building submission..."
# Copy the finalized deck to the root temporarily for the tarball
cp decks/deck.csv ./deck.csv
tar -czvf submission.tar.gz main.py deck.csv src/
# Clean up
rm ./deck.csv
echo "Done. submission.tar.gz created."
