import time 
import random
import asyncio
import numpy as np
from io import BytesIO
from fastapi import FastAPI
import matplotlib.pyplot as plt
from starlette.responses import StreamingResponse


X = np.load("data/The Eiffel Tower.npy", encoding='latin1', allow_pickle=True)

app = FastAPI()

@app.get("/generate/{idx}")
async def generate(idx: int):
    fig = plt.figure()
    try:
        plt.imshow(X[idx].reshape(28, 28), cmap='Greys')
    except ValueError:
        await asyncio.sleep(random.random(0.1))
        plt.imshow(X[idx].reshape(28, 28), cmap='Greys')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(content=buf, media_type="image/png")
