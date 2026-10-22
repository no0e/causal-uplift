"""Uplift modelling and budgeted targeting on the Criteo uplift dataset."""

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ("visit", "conversion")
