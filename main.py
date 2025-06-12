import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

#Person A 
def main():
    print("Hello from lab1-rk!")

    # Load dataset
    import seaborn as sns
    df = sns.load_dataset("penguins").dropna()

    # Encode categorical values
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df['species'] = df['species'].astype('category').cat.codes
    df['island'] = df['island'].astype('category').cat.codes

    # Split dataset
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Dataset split complete.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Person B: build the model
    model = build_default_model()

def build_default_model():
    model = xgb.XGBClassifier()
    print("Default XGBoost model created:", model)
    return model

if __name__ == "__main__":
    main()

