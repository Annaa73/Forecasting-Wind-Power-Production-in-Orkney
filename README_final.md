# Wind Power Forecasting in Orkney

This repository implements a reproducible machine learning pipeline for forecasting wind power production in the Orkney archipelago using historical wind data and weather forecasts.

The project demonstrates the full ML lifecycle including:

* Data preprocessing and feature engineering
* Model training and evaluation
* Experiment tracking with MLflow
* Model registration and serving
* Reproducible training with MLflow Projects

---

# Project Structure

```
.
├── data/
│   ├── power.csv
│   ├── weather.csv
│   └── future.csv
├── plots/
├── script.py
├── helpers.py
├── requirements.txt
├── python_env.yaml
├── MLproject
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the MLflow tracking server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open the MLflow UI in your browser:

```
http://127.0.0.1:5000
```

---

# Running Training

You can run the training pipeline in two ways.

## Option 1 (Direct Python execution)

```bash
python script.py
```

This will:

* Load and align the data
* Perform preprocessing and feature engineering
* Train several regression models
* Perform time-series cross validation
* Log experiments to MLflow
* Select the best model
* Evaluate on the test set
* Generate predictions for future data
* Register the best model in MLflow

---

## Option 2 (Recommended: MLflow Project)

This project is packaged as an **MLflow Project** and can be run reproducibly with:

```bash
mlflow run .
```

MLflow will:

* Create the specified environment
* Install dependencies
* Run the training pipeline automatically

---

# Model Outputs

After training, the following artifacts are generated:

* `plots/eda_plots.png` – exploratory data analysis plots
* `plots/predictions_final.png` – predicted vs actual power output
* `future_predictions_final.csv` – forecasts for future weather data

These artifacts are also logged in **MLflow**.

---

# Model Serving

Once the model is registered in MLflow, it can be served using:

```bash
mlflow models serve -m "models:/WindPowerForecastModelFinal/Production" -p 1234
```

This starts a REST API endpoint.

Example prediction request:

```bash
curl -X POST http://127.0.0.1:1234/invocations \
-H "Content-Type: application/json" \
-d '{
"dataframe_records":[
{
"Speed": 10,
"Direction": "W",
"time": "2023-01-01 12:00:00"
}
]
}'
```

The server will return a predicted wind power output.

---

# Reproducibility

The project uses **MLflow Projects** to ensure reproducibility.

The environment and dependencies are specified in:

* `MLproject`
* `python_env.yaml`

This allows the entire training pipeline to be executed from scratch on another machine.

---

# Author

<Anna Lekston/ awle@itu.dk>