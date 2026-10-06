"""
scripts/generate_traffic.py
============================
Sends N realistic random /predict requests to the API for demo and load testing.

Usage:
    python scripts/generate_traffic.py [--n 100] [--url http://localhost:8000] [--seed 42]
"""

import argparse
import random
import time

import requests

# --------------------------------------------------------------------------
# Sample realistic symptom clusters (reflect real disease presentations)
# --------------------------------------------------------------------------
SYMPTOM_CLUSTERS = [
    # Fungal infection
    ["itching", "skin_rash", "nodal_skin_eruptions"],
    ["itching", "skin_rash"],
    # Allergy
    ["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "runny_nose"],
    ["redness_of_eyes", "throat_irritation", "continuous_sneezing"],
    # GERD / Acid Reflux
    ["acidity", "ulcers_on_tongue", "vomiting", "indigestion"],
    ["stomach_pain", "acidity", "chest_pain"],
    # Chronic cholestasis
    ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite"],
    # Drug Reaction
    ["itching", "skin_rash", "stomach_pain", "vomiting", "bruising"],
    # Peptic ulcer
    ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases"],
    # Diabetes
    ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "polyuria", "increased_appetite", "excessive_hunger"],
    # Hypertension
    ["headache", "dizziness", "chest_pain", "loss_of_balance", "lack_of_concentration"],
    # Migraine
    ["headache", "nausea", "blurred_and_distorted_vision", "stiff_neck", "depression", "visual_disturbances"],
    # Malaria
    ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea", "muscle_pain"],
    # Chicken pox
    ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
    # Dengue
    ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain", "red_spots_over_body"],
    # Typhoid
    ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_(typhos)", "belly_pain"],
    # Hepatitis A
    ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes"],
    # Jaundice
    ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "dark_urine", "abdominal_pain", "yellowing_of_eyes"],
    # Pneumonia
    ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    # Dimorphic Haemorrhoids
    ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    # Heart Attack (EMERGENCY)
    ["vomiting", "sweating", "breathlessness", "chest_pain"],
    ["chest_pain", "breathlessness", "fast_heart_rate"],
    # Urinary tract infection
    ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    # Bronchial Asthma
    ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
    # Arthritis
    ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    # Cervical spondylosis
    ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
    # Paralysis (Blood Clot) - EMERGENCY
    ["vomiting", "headache", "weakness_in_limbs", "weakness_of_one_body_side", "stiff_neck", "altered_sensorium"],
    # Acne
    ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    # Common Cold
    ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose", "congestion", "swelled_lymph_nodes", "malaise", "throat_irritation", "redness_of_eyes", "sinus_pressure"],
    # Impetigo
    ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
    # Psoriasis
    ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
]


def make_predict_request(url: str, symptoms: list, session: requests.Session) -> dict:
    try:
        resp = session.post(f"{url}/predict", json={"symptoms": symptoms}, timeout=10)
        return {"status": resp.status_code, "ok": resp.status_code == 200}
    except requests.RequestException as e:
        return {"status": 0, "ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic for the Disease Prediction API.")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send (default: 100)")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base API URL (default: http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs (default: 42)")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds between requests (default: 0.05)")
    args = parser.parse_args()

    random.seed(args.seed)
    session = requests.Session()

    print(f"Sending {args.n} traffic requests to {args.url} ...\n")

    ok_count = 0
    error_count = 0

    for i in range(1, args.n + 1):
        cluster = random.choice(SYMPTOM_CLUSTERS)
        # Optionally add 0-2 random extra symptoms to vary cardinality
        extra = random.randint(0, 2)
        all_symptoms = [
            "itching", "skin_rash", "vomiting", "fatigue", "headache",
            "nausea", "cough", "chest_pain", "back_pain", "joint_pain",
            "dizziness", "constipation", "abdominal_pain", "high_fever",
        ]
        extra_syms = random.sample(all_symptoms, min(extra, len(all_symptoms)))
        symptoms = list(dict.fromkeys(cluster + extra_syms))  # deduplicate

        result = make_predict_request(args.url, symptoms, session)
        if result["ok"]:
            ok_count += 1
        else:
            error_count += 1

        if i % 10 == 0 or i == args.n:
            pct = int(i / args.n * 50)
            bar = "█" * pct + "░" * (50 - pct)
            print(f"\r[{bar}] {i}/{args.n}  ✓ {ok_count}  ✗ {error_count}", end="", flush=True)

        time.sleep(args.delay)

    print(f"\n\n{'='*60}")
    print("Traffic Generation Complete")
    print(f"  Total requests : {args.n}")
    print(f"  Successful     : {ok_count} ({100*ok_count//args.n}%)")
    print(f"  Failed         : {error_count}")
    print(f"{'='*60}")

    if error_count > 0:
        print("\n⚠  Some requests failed. Check that the API is running at:", args.url)


if __name__ == "__main__":
    main()
