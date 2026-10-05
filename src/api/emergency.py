"""
src/api/emergency.py
====================
Independent emergency triage evaluator.
Reads config/emergency_rules.json and evaluates red-flag symptoms and combinations
independently of the machine learning prediction model.
"""

import os
import json
from typing import List, Tuple, Dict, Any


class EmergencyEvaluator:
    """Evaluates red-flag symptoms and critical combinations against configured clinical rules."""

    def __init__(self, rules_path: str = "config/emergency_rules.json"):
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not os.path.exists(self.rules_path):
            return {
                "critical_individual_symptoms": [],
                "critical_combinations": [],
            }
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, symptoms: List[str]) -> Tuple[bool, List[str]]:
        """
        Evaluate a list of user-provided symptoms against emergency rules.

        Returns:
            (emergency_flag: bool, reasons: List[str])
        """
        symptom_set = {s.strip().lower() for s in symptoms}
        reasons = []

        # Check critical individual red flags
        for item in self.rules.get("critical_individual_symptoms", []):
            crit_sym = item.get("symptom", "").strip().lower()
            if crit_sym and crit_sym in symptom_set:
                reason = item.get("reason", f"Critical red-flag symptom detected: {crit_sym}")
                reasons.append(reason)

        # Check critical symptom combinations
        for combo in self.rules.get("critical_combinations", []):
            combo_symptoms = {s.strip().lower() for s in combo.get("symptoms", [])}
            if combo_symptoms and combo_symptoms.issubset(symptom_set):
                reason = combo.get("reason", f"High-risk symptom combination detected: {list(combo_symptoms)}")
                reasons.append(reason)

        emergency_flag = len(reasons) > 0
        return emergency_flag, reasons
