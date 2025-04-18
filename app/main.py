import numpy as np
from datetime import datetime, timedelta
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI


class ParkingForecaster:
    def __init__(self):
        self.models = {}  # Dictionary to store a model for each parking lot
        self.scalers = {}  # Dictionary to store scalers for each parking lot

    def load_models(self, folder_path):
        """Load trained models and scalers from disk"""
        park_ids = [4,8,9,10,20,29,30,31,46,47,62,67,71,72,73,75,76,77,78,80,81,82,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,101,102,103,104,105,106,108,109,113,114,]
        for park_id in park_ids:
            self.models[park_id] = joblib.load(f"{folder_path}/model_park_{park_id}.pkl")
            self.scalers[park_id] = joblib.load(f"{folder_path}/scaler_park_{park_id}.pkl")

    def predict_trend(self, target_datetime, park_id, recent_data, intervals=[30, 60]):
        """
        Predict parking trends for specific intervals (in minutes)

        Parameters:
        - target_datetime: datetime object for prediction start time
        - park_id: ID of the parking lot to predict for
        - recent_data: DataFrame with recent data including prev_spaces and prev_spaces_2
        - intervals: List of minutes ahead to predict [30, 60] means 30min and 1hr predictions

        Returns:
        - Dictionary with predictions and trends
        """
        # if park_id not in self.models:
        #    raise ValueError(f"No model found for parking lot {park_id}")

        # Extract required features
        hour = target_datetime.hour
        minute = target_datetime.minute
        day_of_week = target_datetime.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Get last two available spaces values for this parking lot
        prev_spaces = recent_data  # recent_data['available_spaces'].iloc[-1]
        prev_spaces_2 = recent_data  # recent_data['available_spaces'].iloc[-2]

        # Create input features
        X_pred = np.array(
            [[hour, minute, day_of_week, is_weekend, prev_spaces, prev_spaces_2]]
        )
        X_pred_scaled = self.scalers[park_id].transform(X_pred)

        # Current prediction
        current_prediction = self.models[park_id].predict(X_pred_scaled)[0]

        results = {
            "park_id": park_id,
            "current_spaces": prev_spaces,
            "current_prediction": current_prediction,
            "forecasts": [],
        }

        # Predict for each interval
        for interval in intervals:
            future_time = target_datetime + timedelta(minutes=interval)
            future_hour = future_time.hour
            future_minute = future_time.minute

            # For future predictions, use predictions as previous values
            X_future = np.array(
                [
                    [
                        future_hour,
                        future_minute,
                        day_of_week,
                        is_weekend,
                        current_prediction,
                        prev_spaces,
                    ]
                ]
            )
            X_future_scaled = self.scalers[park_id].transform(X_future)
            future_prediction = self.models[park_id].predict(X_future_scaled)[0]

            # Calculate trend (positive means more available spaces, negative means fewer)
            trend = future_prediction - prev_spaces
            #trend_percentage = (trend / prev_spaces) * 100 if prev_spaces > 0 else 0

            if trend > 0:
                trend_description = "emptier"
            elif trend < 0:
                trend_description = "fuller"
            else:
                trend_description = "steady"

            results["forecasts"].append(
                {
                    "interval_minutes": interval,
                    "prediction": future_prediction,
                    "absolute_change": trend,
                    #"percentage_change": trend_percentage,
                    "trend": trend_description,
                }
            )

            # Update for next interval prediction
            current_prediction = future_prediction

        return results
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    app.state.forecaster = ParkingForecaster()
    app.state.forecaster.load_models("models/")
    yield
    # Shutdown event (optional cleanup)
    del app.state.forecaster

app = FastAPI(lifespan=lifespan)

@app.get("/api/py/forecast")
def forecast(id: str, current_ava: int):
    """
    Forecast available parking spaces for a given parking lot ID and current available spaces.

    Parameters:
    - id: ID of the parking lot
    - current_ava: Current available spaces in the parking lot

    Returns:
    - JSON response with forecasted values
    """
    # Get the current datetime
    now = datetime.now()

    # Load the forecaster from the app state
    forecaster = app.state.forecaster

    # Predict trends
    prediction = forecaster.predict_trend(now, int(id), current_ava)

    return prediction