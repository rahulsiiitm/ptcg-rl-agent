"""
Build top20_decks/ folder from known meta archetypes.
All decks exactly 60 cards.
"""
import os, csv

id_to_name = {}
with open("data/EN_Card_Data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = int(row["Card ID"])
        id_to_name[cid] = row["Card Name"].strip()

def write_deck(ids, label, filename):
    if len(ids) != 60:
        print(f"  ERROR [{label}]: {len(ids)} cards, need 60!")
        return False
    os.makedirs("top20_decks", exist_ok=True)
    with open(f"top20_decks/{filename}.csv","w") as f:
        f.write("\n".join(str(x) for x in ids)+"\n")
    print(f"  OK [{label}] -> top20_decks/{filename}.csv")
    return True

# 1. Alakazam/Dudunsparce — exact from v15/deck.csv
alak_deck = [5,5,19,19,19,19,65,65,66,66,66,66,305,741,741,741,741,742,742,742,742,743,743,743,743,1079,1079,1079,1081,1081,1081,1086,1086,1086,1086,1097,1097,1097,1129,1152,1152,1152,1152,1182,1182,1182,1184,1197,1197,1197,1225,1225,1225,1225,1227,1231,1231,1231,1231,1247]
write_deck(alak_deck, "Alakazam/Dudunsparce", "alakazam_dudunsparce")

# 2. Grimmsnarl ex — from v15 _GRIMM_IDS exactly
grimm_deck = [7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,647,647,647,648,648,648,860,860,1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,1182,1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259]
write_deck(grimm_deck, "Grimmsnarl ex", "grimmsnarl_ex")

# 3. Great Tusk — from v15 _TUSK_IDS exactly
tusk_deck = [58,58,58,58,344,344,344,344,345,345,1142,1142,1142,1142,1152,1152,1152,1152,1086,1086,1086,1086,1122,1122,1122,1122,1121,1123,1123,1123,1123,1197,1197,1197,1197,1185,1185,1185,1185,1182,1182,1182,1182,1204,1204,1194,1194,1247,1147,20,20,20,20,11,11,11,11,345,345,607]
write_deck(tusk_deck, "Great Tusk", "great_tusk")

# 4. Dragapult ex / Dusknoir (60 exact)
drag_deck = [119,119,119,119,120,120,120,121,121,121,131,131,131,131,132,132,133,133,1079,1079,1079,1079,1086,1086,1086,1086,1097,1097,1097,1152,1152,1152,1152,1182,1182,1182,1227,1227,1227,1231,1231,1247,1159,5,5,5,5,11,11,11,11,13,13,1161,1161,1184,1225,1225,17,17]
write_deck(drag_deck, "Dragapult ex/Dusknoir", "dragapult_ex")

# 5. Lucario Fighting (our deck — exact from deck.csv)
lucario_deck = [673,673,674,674,675,675,676,676,676,677,677,677,678,678,678,678,1102,1102,1102,1102,1123,1123,1141,1141,1141,1141,1142,1142,1142,1142,1152,1152,1152,1152,1159,1182,1182,1192,1192,1192,1192,1227,1227,1227,1227,1252,1252,6,6,6,6,6,6,6,6,6,6,6,6,6]
write_deck(lucario_deck, "Mega Lucario ex", "lucario_fighting")

# 6. Iono Bellibolt ex (60 exact)
belli_deck = [268,268,268,268,269,269,269,270,270,270,270,271,271,271,4,4,4,4,4,4,4,4,4,1086,1086,1086,1086,1079,1079,1079,1079,1121,1121,1121,1224,1224,1224,1224,1182,1182,1182,1125,1174,1152,1152,1152,1152,1227,1227,1231,1231,1097,1097,1097,1159,1097,19,19,19,19]
write_deck(belli_deck, "Iono Bellibolt ex", "iono_bellibolt")

# 7. Raging Bolt ex (60 exact)
raging_deck = [63,63,63,63,37,37,37,27,27,27,344,344,344,345,345,20,20,20,20,20,20,20,11,11,11,11,1086,1086,1086,1086,1097,1097,1097,1152,1152,1152,1152,1182,1182,1182,1227,1227,1227,1225,1225,1247,1159,1159,1081,1081,1156,1156,5,5,5,5,19,19,19,19]
write_deck(raging_deck, "Raging Bolt/Iron Thorns", "raging_bolt_iron_thorns")

# 8. Terapagos ex / Bouffalant (60 exact)
terapagos_deck = [176,176,176,176,234,234,234,233,233,233,631,631,631,173,173,173,174,174,1250,1250,1250,1250,1086,1086,1086,1086,1097,1097,1097,1152,1152,1152,1152,1182,1182,1182,1227,1227,1227,1247,1247,1159,1159,1079,1079,1121,1121,1142,1142,5,5,5,5,19,19,19,19,11,11,11]
write_deck(terapagos_deck, "Terapagos/Bouffalant", "terapagos_bouffalant")

# 9. Ethan's Typhlosion (60 exact)
typhlosion_deck = [354,354,354,354,352,352,352,352,353,353,1079,1079,1079,1079,1086,1086,1086,1086,1097,1097,1097,1152,1152,1152,1152,1182,1182,1182,1227,1227,1227,1231,1231,1247,1159,1159,1225,1225,1142,1142,5,5,5,5,19,19,19,19,17,17,17,17,11,11,11,11,2,2,2,2]
write_deck(typhlosion_deck, "Ethan's Typhlosion", "typhlosion")

print("\n=== Summary ===")
total = 0
for fname in sorted(os.listdir("top20_decks")):
    lines = open(f"top20_decks/{fname}").read().strip().split("\n")
    total += 1
    print(f"  {fname}: {len(lines)} cards")
print(f"\n{total} deck templates ready.")
