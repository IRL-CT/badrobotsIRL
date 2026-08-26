import os
import csv
import json
from collections import defaultdict

# 1. Exact Video Durations
durations = {
    'p10nodbot': 41.012, 'p11nodbot': 36.016, 'p12nodbot': 45.016, 'p14nodbot': 35.033,
    'p16nodbot': 45.027, 'p17nodbot': 38.018, 'p18nodbot': 43.027, 'p19nodbot': 39.027,
    'p20nodbot': 45.013, 'p21nodbot': 40.016, 'p22nodbot': 37.015, 'p23nodbot': 40.025,
    'p24nodbot': 43.017, 'p25nodbot': 38.006, 'p26nodbot': 38.642, 'p27nodbot': 52.014,
    'p28nodbot': 38.025, 'p29nodbot': 55.026, 'p2nodbot': 39.002, 'p3nodbot': 85.033,
    'p4nodbot': 33.019, 'p5nodbot': 39.019, 'p6nodbot': 40.007, 'p7nodbot': 38.032,
    'p8nodbot': 34.002, 'p9nodbot': 35.030
}

# 2. Definitive Door Open and Leakage Updates
user_updates = {
    'p2nodbot': [('researcher', 36.0, durations['p2nodbot'])],
    'p4nodbot': [('researcher', 29.0, durations['p4nodbot'])],
    'p6nodbot': [('researcher', 34.0, durations['p6nodbot'])],
    'p7nodbot': [('researcher', 32.0, durations['p7nodbot'])],
    'p8nodbot': [
        ('nodbot', 20.0, 21.0), # Leaked 'Sorry'
        ('researcher', 29.0, durations['p8nodbot'])
    ],
    'p9nodbot': [('researcher', 29.0, durations['p9nodbot'])],
    'p10nodbot': [('researcher', 35.0, durations['p10nodbot'])],
    'p11nodbot': [('researcher', 33.0, durations['p11nodbot'])],
    'p12nodbot': [
        ('nodbot', 34.0, 37.0), # Leaked 'Okay'
        ('researcher', 40.0, durations['p12nodbot'])
    ],
    'p14nodbot': [('researcher', 30.0, durations['p14nodbot'])],
    'p16nodbot': [('researcher', 40.0, durations['p16nodbot'])],
    'p17nodbot': [('researcher', 33.0, durations['p17nodbot'])],
    'p18nodbot': [('researcher', 37.0, durations['p18nodbot'])],
    'p19nodbot': [('researcher', 34.0, durations['p19nodbot'])],
    'p20nodbot': [('researcher', 39.0, durations['p20nodbot'])],
    'p21nodbot': [('researcher', 35.0, durations['p21nodbot'])],
    'p22nodbot': [('researcher', 32.0, durations['p22nodbot'])],
    'p23nodbot': [('researcher', 33.0, durations['p23nodbot'])],
    'p25nodbot': [('researcher', 31.0, durations['p25nodbot'])],
    'p26nodbot': [('researcher', 34.0, durations['p26nodbot'])],
    'p27nodbot': [('researcher', 47.0, durations['p27nodbot'])],
    'p28nodbot': [('researcher', 31.0, durations['p28nodbot'])],
    'p29nodbot': [('researcher', 48.0, durations['p29nodbot'])],
}

def apply_updates():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_file = os.path.join(base_dir, "HRI25_LBR", "data", "speaker_diarization_all_audio.csv")
    cleaned_file = os.path.join(base_dir, "HRI25_LBR", "data", "speaker_diarization_cleaned.csv")
    json_file = os.path.join(base_dir, "preprocessing", "diarization_data.json")

    # Read existing rows, filtering out previously injected researcher/override rows
    rows_by_participant = defaultdict(list)
    with open(orig_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            spk = r["speaker"].strip().lower()
            # Ignore any prior researcher rows so re-running doesn't create duplicate entries
            if spk in ["researcher", "door_open"]:
                continue
            rows_by_participant[r["participant_num"]].append({
                "participant_num": r["participant_num"],
                "speaker": spk,
                "start": float(r["start"]),
                "end": float(r["end"]),
                "duration": float(r["duration"])
            })

    # Apply updates cleanly
    all_cleaned_rows = []
    
    for p_num in sorted(rows_by_participant.keys()):
        existing = rows_by_participant[p_num]
        updates = user_updates.get(p_num, [])

        for spk, s, e in updates:
            dur = round(e - s, 3)
            # Avoid adding if exact same range exists
            if not any(x['speaker'] == spk and abs(x['start'] - s) < 0.05 for x in existing):
                existing.append({
                    "participant_num": p_num,
                    "speaker": spk,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "duration": dur
                })

        # Deduplicate and sort by start time
        existing.sort(key=lambda x: x["start"])
        all_cleaned_rows.extend(existing)

    fieldnames = ["participant_num", "speaker", "start", "end", "duration"]
    for out_path in [cleaned_file, orig_file]:
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_cleaned_rows:
                writer.writerow(r)
        print(f"Saved {len(all_cleaned_rows)} rows to {out_path}")

    # Update JSON for web player
    json_data = defaultdict(list)
    for r in all_cleaned_rows:
        json_data[r["participant_num"]].append({
            "speaker": r["speaker"],
            "start": r["start"],
            "end": r["end"],
            "duration": r["duration"]
        })

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Updated {json_file}")

if __name__ == "__main__":
    apply_updates()
