########################################################################################################################
# IMPORTS
# You absolutely need these
import mlflow
import os

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from mlflow.models import infer_signature

### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from helpers import FeatureEngineering, InterpolateData, Imputer

########################################################################################################################


def align_and_resample(wind_df, power_df, freq="3H"):
    # Downsample power to given frequency
    power_resampled = power_df.set_index("time").resample(freq).mean().reset_index()
    # Merge with wind data
    data = pd.merge(wind_df, power_resampled, on="time", how="inner")
    return data

def eda_visualizations(data):
    data_viz = data.copy()

    # Subplots
    fig, ax = plt.subplots(1,3, figsize=(25,4))

    # Speed and Power for the last 7 days
    data_viz["hour"] = data_viz["time"].dt.hour
    hourly = data_viz.groupby("hour")["Total"].mean()
    ax[0].plot(hourly.index, hourly.values, marker="o", label="Average power")
    ax[0].set_title("Average wind power by hour")
    ax[0].set_xlabel("Hour of day")
    ax[0].tick_params(axis='x', labelrotation = 45)
    ax[0].set_ylabel("Average power (MW)")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(data["Speed"], data["Total"])
    power_curve = data.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Power curve (binned wind speed vs average power)
    data_viz["speed_bin"] = pd.cut(data_viz["Speed"], bins=20)
    power_curve = data_viz.groupby("speed_bin")["Total"].mean()
    ax[2].plot(power_curve.index.astype(str), power_curve.values, marker="o")
    ax[2].set_title("Wind Turbine Power Curve (Binned)")
    ax[2].set_xlabel("Wind Speed Bin (m/s)")
    ax[2].set_ylabel("Average Power Output (MW)")
    ax[2].tick_params(axis="x", rotation=45)
    ax[2].grid(True, linestyle="--", alpha=0.6)

    return fig


def main():
    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("WindPowerForecastingFinal")

    # Load datasets
    power_df = pd.read_csv('data/power.csv', parse_dates=["time"])
    wind_df = pd.read_csv('data/weather.csv', parse_dates=["time"])
    future_data = pd.read_csv("data/future.csv", parse_dates=["time"])

    # Align and resample
    data = align_and_resample(wind_df, power_df)
    
    # Sort by time to preserve temporal order
    data = data.sort_values("time").reset_index(drop=True)

    # --- Create and save EDA plots ---
    os.makedirs("plots", exist_ok=True)
    eda_fig = eda_visualizations(data)
    eda_fig.savefig("plots/eda_plots.png")
    plt.close(eda_fig)

    # Log EDA once in its own run
    with mlflow.start_run(run_name="EDA"):
        mlflow.log_artifact("plots/eda_plots.png")

    # remove missing "Total" values
    data = data.dropna(subset=["Total"])

    # Hold out final test set (20%) 
    split_idx = int(len(data) * 0.8)
    train_val_df = data.iloc[:split_idx]
    test_df      = data.iloc[split_idx:]

    # Separate features / target
    X_train = train_val_df.drop(columns=["Total"])
    y_train = train_val_df["Total"]

    X_test  = test_df.drop(columns=["Total"])
    y_test  = test_df["Total"]
    
    # Define models and hyperparameters 
    experiments = {
        "RandomForest": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        },
        "KNN": {
            "model": KNeighborsRegressor,
            "params": {
                "n_neighbors": [3, 5]
            }
        },
        "LinearRegression": {
            "model": LinearRegression,
            "params": {}
        }
    }
    
    # Time Series CV 
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    best_score = np.inf
    best_model_info = None
    
    # Grid search over models & hyperparameters - CV
    for model_name, config in experiments.items():
        for params in ParameterGrid(config["params"]):

            model = config["model"](**params)
                            
            # Pipeline 
            pipe = Pipeline([
                ("features", FeatureEngineering()),
                ("interpolate", InterpolateData()),
                ("imputer", Imputer()),
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            
            # CV Metrics
            fold_mae = []
            fold_mse = []
            fold_rmse = []
            fold_r2 = []
            fold_ev = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_val)
                    
                fold_mae.append(mean_absolute_error(y_val, preds))
                fold_mse.append(mean_squared_error(y_val, preds))
                fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
                fold_r2.append(r2_score(y_val, preds))
                fold_ev.append(explained_variance_score(y_val, preds))
                
            # Average CV metrics 
            cv_metrics = {
                "MAE": np.mean(fold_mae),
                "MSE": np.mean(fold_mse),
                "RMSE": np.mean(fold_rmse),
                "R2": np.mean(fold_r2),
                "ExplainedVariance": np.mean(fold_ev)
            }
                
            # Log to MLflow
            with mlflow.start_run(run_name=f"{model_name}_{params}"):
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                for metric_name, metric_val in cv_metrics.items():
                    mlflow.log_metric(f"CV_{metric_name}", metric_val)

            # Track results 
            results[f"{model_name}_{params}"] = cv_metrics

            # Select best model (based on RMSE)
            if cv_metrics["RMSE"] < best_score:
                best_score = cv_metrics["RMSE"]
                best_model_info = (model_name, params)

    print(f"Best model: {best_model_info} with CV_RMSE={best_score:.3f}")

    # Train only the best model on full training data 
    best_model_name, best_params = best_model_info
    best_model = experiments[best_model_name]["model"](**best_params)

    best_pipeline = Pipeline([
        ("features", FeatureEngineering()),
        ("interpolate", InterpolateData()),
        ("imputer", Imputer()),
        ("scaler", StandardScaler()),
        ("model", best_model)
    ])

    best_pipeline.fit(X_train, y_train)        
                
    # Evaluate on held-out test set 
    test_preds = best_pipeline.predict(X_test)

    test_metrics = {
        "MAE": mean_absolute_error(y_test, test_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
        "R2": r2_score(y_test, test_preds),
        "ExplainedVariance": explained_variance_score(y_test, test_preds)
    }

    print("Test metrics:", test_metrics)

    # Plot predictions
    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(test_preds)), test_preds, label="Predictions")
    plt.plot(np.arange(len(y_test)), y_test, label="Truth")
    plt.legend()
    plt.savefig(f"plots/predictions_final.png")
    plt.close()
                
    # Forecast on future dataset 
    future_preds = best_pipeline.predict(future_data)
    future_data["Total_Predicted"] = future_preds
    future_data.to_csv("future_predictions_final.csv", index=False)

    # Log the best model (trained on the full training dataset)
    with mlflow.start_run(run_name="Best_Model"):

        mlflow.log_param("model_type", best_model_name)
        mlflow.log_params(best_params)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # infer model signature
        signature = infer_signature(X_train, best_pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            best_pipeline,
            "model",
            signature=signature,
            registered_model_name="WindPowerForecastModelFinal"
        )

        mlflow.log_artifact("plots/predictions_final.png")
        mlflow.log_artifact("future_predictions_final.csv")

    print("Forecasting completed! Predictions saved to future_predictions.csv")

if __name__ == "__main__":
    main()