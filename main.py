from fastapi import FastAPI, HTTPException
from io import BytesIO
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionPipeline
import base64


app = FastAPI()


# Mocking the model here
class MockModel:
    def __init__(self):
        self.loaded = False
        self.pipe = None

    def load(self):
        self.loaded = True
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_auth_token="False",
        )
        self.pipe = self.pipe.to("cuda:0")

    def unload(self):
        self.loaded = False
        self.pipe = None

    def infer(self, text):
        # Generate a simple image with the given text
        image = self.pipe(text).images[0]
        return image


model = MockModel()
model.load()


@app.get("/v2")
def health_check():
    return {"model-name": "stable-diffusion"}


@app.get("/v2/health/live")
def health_check():
    return {"status": "running"}


@app.get("/v2/health/ready")
def health_check():
    return {"status": "running"}


@app.get("/v2/models/stable-diffusion/ready")
@app.get("/v2/models/stable-diffusion/versions/1/ready")
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
def generate_image(text: str):
    if not model.loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded")

    image = model.infer(text)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return {"image": img_str.decode('utf-8')}
