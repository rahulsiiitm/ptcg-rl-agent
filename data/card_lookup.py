import csv
import os
from collections import namedtuple

# Standard columns based on the CSV structure:
# 0: Card ID, 1: Card Name, 2: Expansion, 3: Collection No., 4: Stage/Type, 5: Rule, 
# 6: Category, 7: Previous stage, 8: HP, 9: Type, 10: Weakness, 11: Resistance, 
# 12: Retreat, 13: Move Name, 14: Cost, 15: Damage, 16: Effect Explanation

Card = namedtuple("Card", [
    "id", "name", "expansion", "collection_no", "stage", "rule", 
    "category", "previous_stage", "hp", "type", "weakness", 
    "resistance", "retreat", "move_name", "cost", "damage", "effect"
])

class CardLookup:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Default to the same directory as this script
            csv_path = os.path.join(os.path.dirname(__file__), "EN_Card_Data.csv")
            
        self.cards = {}
        self.load_data(csv_path)

    def load_data(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                # Some rows might not have all 17 columns if trailing commas are missing, so pad it
                row = row + [''] * (17 - len(row))
                try:
                    card_id = int(row[0])
                except ValueError:
                    continue
                
                # If a card has multiple moves, it might appear multiple times or not, 
                # but let's just store a list of moves per card ID or keep it simple.
                # Actually, the dataset has 1 row per move, so let's store a list of rows per card.
                if card_id not in self.cards:
                    self.cards[card_id] = []
                
                self.cards[card_id].append(Card(*row[:17]))

    def get_card(self, card_id: int):
        """Returns all rows associated with a card ID."""
        return self.cards.get(card_id, [])

    def filter_cards(self, **kwargs):
        """
        Filter cards by attributes (e.g. stage='Basic', rule='n/a', type='{W}').
        Returns a dict of card_id -> list of Card tuples.
        """
        results = {}
        for cid, rows in self.cards.items():
            first_row = rows[0]
            match = True
            for k, v in kwargs.items():
                attr_val = getattr(first_row, k, None)
                if attr_val is None or attr_val != v:
                    match = False
                    break
            if match:
                results[cid] = rows
        return results

if __name__ == "__main__":
    db = CardLookup()
    print(f"Loaded {len(db.cards)} unique cards.")
