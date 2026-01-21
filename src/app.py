from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import torch
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="DQN Atari Agent")

class StateInput(BaseModel):
    state: List[List[List[List[float]]]]

class PredictionOutput(BaseModel):
    action: int
    q_values: List[float]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: StateInput):
    try:
        state = np.array(input_data.state, dtype=np.float32)
        if state.shape != (1, 4, 84, 84):
            raise ValueError(f"Invalid state shape: {state.shape}")
        
        # Load model and predict
        from src.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=(4, 84, 84), action_dim=6)
        
        q_values = agent.get_q_values(state)
        action = np.argmax(q_values)
        
        return PredictionOutput(
            action=int(action),
            q_values=q_values[0].tolist()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
