#!/bin/bash
# Builds the submission.tar.gz for Kaggle
# Includes: agent, deck, cg module, card data, archetype templates, weight overrides

echo "Building submission..."
tar -czvf submission.tar.gz \
    main.py \
    deck.csv \
    lucario_w.json \
    src/ \
    cg/ \
    data/EN_Card_Data.csv \
    data/card_lookup.py \
    data/__init__.py
echo "Done. submission.tar.gz created."
echo "Size: $(du -sh submission.tar.gz | cut -f1)"
