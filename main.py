from fastapi import FastAPI, HTTPException
import os
os.environ['HF_HOME'] = "/data/models/hf_cache"
from io import BytesIO
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionPipeline
from transformers import Pipeline
import base64
from pydantic import BaseModel
import time

app = FastAPI()

# Mocking the model here
class MockModel:
    def __init__(self):
        self.loaded = False
        self.pipe = None

    def load(self):
        self.loaded = True
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map='balanced'
        )

    def unload(self):
        self.loaded = False
        self.pipe = None

    def infer(self, text):
        # Generate a simple image with the given text
        image = self.pipe(text).images[0]
        return image

start = time.time()
model = MockModel()
model.load()
print("Time taken" + str( time.time() - start) )

class InferRequest(BaseModel):
    text: str


@app.get("/v2")
@app.get(f"/v2/models/stable-diffusion")
def version():
    return {"name": "stable-diffusion"}


@app.get("/v2/health/live")
def health_check():
    return {"status": "running"}


@app.get("/v2/health/ready")
def health_ready():
    return {"status": "running"}


@app.get("/v2/health/live")
@app.get("/v2/health/ready")
@app.get("/v2/models/stable-diffusion/ready")
@app.get("/v2/models/stable-diffusion/versions/1/ready")
@app.get("/v2/health/live")
def health_check():
    return {"status": "running"}


@app.post("/v2/models/stable-diffusion/load")
def load_model():
    if not model.loaded:
        model.load()
        return {"status": "model loaded"}
    else:
        raise HTTPException(status_code=400, detail="Model is already loaded")


@app.post("/v2/models/stable-diffusion/unload")
def unload_model():
    if model.loaded:
        model.unload()
        return {"status": "model unloaded"}
    else:
        raise HTTPException(status_code=400, detail="Model is not loaded")


@app.post("/v2/models/stable-diffusion/versions/1/infer")
@app.post("/v2/models/stable-diffusion/infer")
def generate_image(request: InferRequest):
    if not model.loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    text = request.text
    image = model.infer(text)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return {"image": img_str.decode('utf-8')}
