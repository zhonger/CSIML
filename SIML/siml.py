import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SIML:
    # Basic attributes
    data = pd.DataFrame()

    X = pd.DataFrame()
    y = pd.DataFrame()

    features = ["Number", "AtomicWeight", "Period", "Group", "Family",
                "LQuantumNumber", "MendeleevNumber",
                "AtomicRadius", "CovalentRadius", "ZungerRadius", "IonicRadius", "CrystalRadius",
                "Electronegativity", "MartynovBatsanovEN", "GordyEN", "MullikenEN", "AllenEN",
                "MetallicValence",
                "NValence", "NsValence", "NpValence", "NdValence", "NUnfilled",
                "FirstIonizationEnergy", "Polarizability",
                "MeltingT", "BoilingT", "Density",
                "SpecificHeat", "HeatFusion", "HeatVaporization", "ThermalConductivity", "HeatAtomization",
                "CohesiveEnergy"]
    stats = ["maximum", "minimum", "avg_dev", "mean"]

    majority_size = {"Total": 0, "Train": 0, "Validation": 0, "Test": 0}
    minority_size = {"Total": 0, "Train": 0, "Validation": 0, "Test": 0}
    majority_set = []
    minority_set = []
    threshold = 0
    majority_pro = 0.2
    minority_pro = 0.2
    random_state = 16432
    ss1 = ""
    ss2 = ""

    method = ""  # basis1, basis2, siml
    basic_model = "SVR"  # SVR, RF, DT
    cv_method = ""

    # Parameters for SVR with RBF kernel
    gamma = 0.01
    C = 10

