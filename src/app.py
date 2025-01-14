import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load data
    total_data = pd.read_csv(
        "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv", sep=","
    )
    print("Initial Data Preview:")
    print(total_data.head())

    # Drop duplicates
    total_data = total_data.drop_duplicates().reset_index(drop=True)
    print("\nData After Removing Duplicates:")
    print(total_data.head())

    # Identify numeric columns
    numeric_columns = [
        col for col in total_data.select_dtypes(include=["float64", "int64"]).columns if col != "Heart disease_number"
    ]
    print("\nNumeric Columns Identified:")
    print(numeric_columns)

    # Split data into train and test sets
    X = total_data[numeric_columns]
    y = total_data["Heart disease_number"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining and Testing Sets Created:")
    print(f"Training Set Size: {len(X_train)}, Testing Set Size: {len(X_test)}")

    # Scale data after split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for compatibility
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("\nData After Scaling:")
    print("Scaled Training Data (First 5 Rows):")
    print(X_train.head())

    # Feature selection using SelectKBest
    k = int(len(X_train.columns) * 0.3)  # Selecting top 30% features
    selection_model = SelectKBest(score_func=f_regression, k=k)
    X_train_sel = selection_model.fit_transform(X_train, y_train)
    X_test_sel = selection_model.transform(X_test)

    # Retrieve selected feature names
    selected_features = X_train.columns[selection_model.get_support()]
    X_train_sel = pd.DataFrame(X_train_sel, columns=selected_features, index=X_train.index)
    X_test_sel = pd.DataFrame(X_test_sel, columns=selected_features, index=X_test.index)

    print("\nSelected Features (Top 30%):")
    print(selected_features)

    # Add target back to selected features
    X_train_sel["Heart disease_number"] = y_train.values
    X_test_sel["Heart disease_number"] = y_test.values

    # Save clean data
    X_train_sel.to_csv("clean_train.csv", index=False)
    X_test_sel.to_csv("clean_test.csv", index=False)

    print("\nClean Data Saved as CSV:")
    print("Clean Training Data (First 5 Rows):")
    print(X_train_sel.head())
    print("\nClean Testing Data (First 5 Rows):")
    print(X_test_sel.head())

    # Reload cleaned datasets
    train_data = pd.read_csv("clean_train.csv")
    test_data = pd.read_csv("clean_test.csv")

    # Split into features and target
    X_train = train_data.drop(columns=["Heart disease_number"])
    y_train = train_data["Heart disease_number"]
    X_test = test_data.drop(columns=["Heart disease_number"])
    y_test = test_data["Heart disease_number"]

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    print("\nLogistic Regression Results:")
    print(f"Intercept: {logistic_model.intercept_[0]}")
    print(f"Coefficients: {logistic_model.coef_}")
    y_pred = logistic_model.predict(X_test)
    print(f"Predicted Values (First 10): {y_pred[:10]}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    # Lasso Regression
    lasso_model = Lasso(alpha=1.0, max_iter=10000)
    lasso_model.fit(X_train, y_train)

    print("\nLasso Regression Results:")
    print(f"Coefficients: {lasso_model.coef_}")
    print(f"R2 Score: {lasso_model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    main()
