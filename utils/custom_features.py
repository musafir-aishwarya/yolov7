import cv2
import time
import torch
import numpy as np
from PIL import Image
from diffusers import LDMSuperResolutionPipeline


def crop_bbox(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cropped_img = img[c1[1]:c2[1], c1[0]:c2[0]]
    return cropped_img


class Latent:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Paper: High-Resolution Image Synthesis with Latent Diffusion Models
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model
        self.pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        self.pipeline = self.pipeline.to(device)

    def inference(self, target_img, step=10):
        print(target_img.shape)
        start_time = time.time()
        # Convert img from BGR to RGB
        upscaled_image = Image.fromarray(
            cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        # Run pipeline in inference (sample random noise and denoise)
        upscaled_image = self.pipeline(
            upscaled_image, num_inference_steps=step, eta=1).images[0]
        # Convert img from RGB back to BGR
        upscaled_image = cv2.cvtColor(
            np.asarray(upscaled_image), cv2.COLOR_RGB2BGR)
        stop_time = time.time()

        print(f'SR inference time: {stop_time-start_time}')
        return upscaled_image
