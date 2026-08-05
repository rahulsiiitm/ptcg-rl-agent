import csv

name_to_id = {}
id_to_name = {}
with open('data/EN_Card_Data.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = int(row['Card ID'])
        name = row['Card Name'].strip()
        name_to_id[name.lower()] = cid
        id_to_name[cid] = name

# Key IDs referenced in v15
v15_ids = [741,742,743,65,305,66,140,142,343,858,174,
           1079,1081,1086,1097,1103,1129,1146,1152,1123,
           1156,1159,1161,1174,1182,1184,1186,1197,1225,1227,1231,
           1244,1246,1247,1264,1266,5,11,13,19,20]
print("=== v15 Alakazam deck card IDs ===")
for cid in v15_ids:
    print(f"  {cid}: {id_to_name.get(cid, 'NOT FOUND')}")

# Key IDs referenced in our deck
our_ids = [673,674,675,676,677,678,344,345,1102,1123,1141,1142,1152,1159,1182,1192,1227,1252,6]
print("\n=== Our Lucario deck card IDs ===")
for cid in sorted(set(our_ids)):
    print(f"  {cid}: {id_to_name.get(cid, 'NOT FOUND')}")

# Print all unique names for search
print("\n=== Searching for key meta cards ===")
keywords = ['grimmsnarl', 'dragapult', 'tusk', 'munkidori', 'iron', 'flutter', 'raging', 'charizard']
for kw in keywords:
    matches = [(cid, name) for name, cid in name_to_id.items() if kw in name]
    if matches:
        for cid, name in sorted(matches):
            print(f"  {cid}: {id_to_name[cid]}")
