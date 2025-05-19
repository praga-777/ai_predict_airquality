import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv("data/clean_air_quality_data.csv")

    X = df.drop(columns=["AQI_Category", "Combined_Pollutant"])
    y = df["AQI_Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "model/air_quality_model.pkl")
    print("Model trained and saved.")

if __name__ == "__main__":
    train_and_save_model()