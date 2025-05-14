# model_creation.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Synthetic TNBC-like dataset
np.random.seed(42)
n = 500
data = {
    "age": np.random.randint(20, 70, n),
    "race_black": np.random.binomial(1, 0.4, n),
    "family_history": np.random.binomial(1, 0.3, n),
    "early_menstruation": np.random.binomial(1, 0.25, n),
    "late_menopause": np.random.binomial(1, 0.2, n),
    "obesity": np.random.binomial(1, 0.35, n),
    "smoking": np.random.binomial(1, 0.25, n),
    "breast_pain": np.random.binomial(1, 0.3, n),
    "palpable_lump": np.random.binomial(1, 0.5, n),
    "nipple_discharge": np.random.binomial(1, 0.2, n),
    "skin_dimpling": np.random.binomial(1, 0.1, n)
}
df = pd.DataFrame(data)
df["tnbc_risk"] = ((df["race_black"] == 1) & (df["palpable_lump"] == 1) & 
                   (df["early_menstruation"] == 0) & (df["late_menopause"] == 0) & 
                   (df["age"] < 50)).astype(int)
X = df.drop("tnbc_risk", axis=1)
y = df["tnbc_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier().fit(X_train, y_train)
joblib.dump(model, "tnbc_risk_model.joblib")
print("Model saved as tnbc_risk_model.joblib")
