import pandas as pd
from datetime import datetime
import joblib, os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class Forecaster:
    def __init__(self):
        self.model_30 = {}
        self.model_60 = {}

    def load(self):
        park_ids = [4,8,9,10,20,29,30,31,46,47,62,67,71,72,73,75,76,77,78,80,81,82,84,85,86,87,88,89,90,92,93,94,95,96,97,98,99,101,102,103,104,105,106,108,109,113,114,]
        base_path = os.path.dirname(__file__)
        for park_id in park_ids:
            load_30 = joblib.load(os.path.join(base_path, f"models/model_park_{park_id}_30.pkl"))
            load_60 = joblib.load(os.path.join(base_path, f"models/model_park_{park_id}_60.pkl"))
            self.model_30[park_id] = load_30
            self.model_60[park_id] = load_60

    def predict_trend(self, park_id: int, current_free_spaces: int):
        current_timestamp = datetime.now()
        hour = current_timestamp.hour
        weekday = current_timestamp.weekday()
        input_data = pd.DataFrame([[hour, weekday, current_free_spaces]], columns=["hour", "weekday", "free_spaces"])

        pred_30 = self.model_30[park_id].predict(input_data)[0]
        pred_60 = self.model_60[park_id].predict(input_data)[0]

        change_30 = round(pred_30 - current_free_spaces)
        change_60 = round(pred_60 - current_free_spaces)

        trend_30 = "emptier" if change_30 > 0 else ("fuller" if change_30 < 0 else "steady")
        trend_60 = "emptier" if change_60 > 0 else ("fuller" if change_60 < 0 else "steady")

        results = {
            "park_id": park_id,
            "current_spaces": current_free_spaces,
            "trend30": trend_30,
            "trend60": trend_60,
            "change30": change_30,
            "change60": change_60,
        }

        return results

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecaster = Forecaster()
    app.state.forecaster.load()
    yield
    del app.state.forecaster

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://next-parking.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/forecast")
def forecast(park_id: int, current_free_spaces: int):
    forecaster = app.state.forecaster
    prediction = forecaster.predict_trend(park_id, current_free_spaces)
    return prediction