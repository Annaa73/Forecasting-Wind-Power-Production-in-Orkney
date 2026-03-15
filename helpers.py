# helpers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        compass_to_deg = { "N":0,"NNE":22.5,"NE":45,"ENE":67.5,
                           "E":90,"ESE":112.5,"SE":135,"SSE":157.5,
                           "S":180,"SSW":202.5,"SW":225,"WSW":247.5,
                           "W":270,"WNW":292.5,"NW":315,"NNW":337.5 }
        if "Direction" in X.columns:
            X["angle"] = X["Direction"].map(compass_to_deg)
            X["angle"] = np.deg2rad(X["angle"])
            X["u"] = -X["Speed"] * np.sin(X["angle"])
            X["v"] = -X["Speed"] * np.cos(X["angle"])
        if "Speed" in X.columns:
            X["wind_speed_cubed"] = X["Speed"]**3
        if "time" in X.columns:
            hour = pd.to_datetime(X["time"]).dt.hour
            X["hour_sin"] = np.sin(2*np.pi*hour/24)
            X["hour_cos"] = np.cos(2*np.pi*hour/24)
        drop_cols = ["Direction", "Speed", "angle", "time", "Lead_hours", "Source_time", "ANM", "Non-ANM"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        return X

class InterpolateData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X)
        return X.interpolate(method="linear", limit_direction="both")

class Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = X.mean(numeric_only=True)
        return self
    def transform(self, X):
        X = X.copy()
        return X.fillna(self.means_)