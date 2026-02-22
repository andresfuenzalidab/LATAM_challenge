import fastapi
from fastapi import HTTPException, Request
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path
from challenge.model import DelayModel
import os

app = fastapi.FastAPI()

_model = None

def get_model():
    global _model # To avoid the model to be loaded every time the endpoint is called, we use a global variable.
    if _model is None:
        _model = DelayModel()
        project_root = Path(__file__).parent.parent
        data_path = os.path.join(project_root, "data", "data.csv")
        data = pd.read_csv(data_path)
        
        features, target = _model.preprocess(data, target_column="delay")
        _model.fit(features, target)
    return _model

def validate_flight(flight: Dict[str, Any],
                    valid_operas = [
                    "Aerolineas Argentinas",
                    "Aeromexico",
                    "Air Canada",
                    "Air France",
                    "Alitalia",
                    "American Airlines",
                    "Austral",
                    "Avianca",
                    "British Airways",
                    "Copa Air",
                    "Delta Air",
                    "Gol Trans",
                    "Grupo LATAM",
                    "Iberia",
                    "JetSmart SPA",
                    "K.L.M.",
                    "Lacsa",
                    "Latin American Wings",
                    "Oceanair Linhas Aereas",
                    "Qantas Airways",
                    "Sky Airline",
                    "United Airlines"
                    ]) -> None:
    if "OPERA" not in flight or "TIPOVUELO" not in flight or "MES" not in flight:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if flight["OPERA"] not in valid_operas:
        raise HTTPException(status_code=400, detail="Invalid OPERA")
    
    if flight["TIPOVUELO"] not in ["I", "N"]:
        raise HTTPException(status_code=400, detail="Invalid TIPOVUELO")
    
    if not isinstance(flight["MES"], int) or flight["MES"] < 1 or flight["MES"] > 12:
        raise HTTPException(status_code=400, detail="Invalid MES")

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(req: Request) -> dict:
    try:
        body = await req.json()
        
        if "flights" not in body or not isinstance(body["flights"], list) or len(body["flights"]) == 0:
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        for flight in body["flights"]:
            validate_flight(flight)
        
        flights_data = body["flights"]
        df = pd.DataFrame(flights_data)
        
        df['Fecha-I'] = '2023-01-01 12:00:00'
        
        model = get_model()
        features = model.preprocess(df)
        predictions = model.predict(features)
        
        return {"predict": predictions}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")