'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import shutil
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

SAFETY_MODEL_ID = "nitrosocke/Ghibli-Diffusion"
MODEL_CACHE_DIR = "diffusers-cache"

def download_model():
    '''
    Downloads the model from the URL passed in.
    '''
    
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    StableDiffusionPipeline.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=model_cache_path,
    )

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    download_model()
