from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc
import sys
from datetime import datetime

model_dir = os.getenv("SDXL_MODEL_DIR")
access_token = os.getenv("ACCESS_TOKEN")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "stabilityai/stable-diffusion-xl-base-1.0")
    model_key_refiner = os.path.join(model_dir, "stabilityai/stable-diffusion-xl-refiner-1.0")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Use refiner (enabled by default)
enable_refiner = os.getenv("ENABLE_REFINER", "true").lower() == "true"
# Output images before the refiner and after the refiner
output_images_before_refiner = True

# Create public link
share = os.getenv("SHARE", "false").lower() == "true"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_auth_token=access_token)

#pipe.enable_model_cpu_offload()
pipe.to("cuda")

# if using torch < 2.0
pipe.enable_xformers_memory_efficient_attention()

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_auth_token=access_token)
    #pipe_refiner.enable_model_cpu_offload()
    pipe_refiner.to("cuda")

    # if using torch < 2.0
    pipe_refiner.enable_xformers_memory_efficient_attention()

    # pipe_refiner.unet = torch.compile(pipe_refiner.unet, mode="reduce-overhead", fullgraph=True)

# NOTE: we do not have word list filtering in this gradio demo

is_gpu_busy = False

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def infer(prompt, negative, scale, samples=4, steps=50, refiner_strength=0.3, num_images=1):
    prompt, negative = [prompt] * samples, [negative] * samples
    images_b64_list = []

    for i in range(0, num_images):
        print("Generating "+str(i+1)+" of "+str(num_images))
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps).images
        os.makedirs(r"/workspace/sdxloutput", exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        
		
        if enable_refiner:
            if output_images_before_refiner:
                for image in images:
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    image_b64 = (f"data:image/jpeg;base64,{img_str}")
                    images_b64_list.append(image_b64)

            images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images, num_inference_steps=steps, strength=refiner_strength).images

            gc.collect()
            torch.cuda.empty_cache()

        # Create the outputs folder if it doesn't exist
        

        for i, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_b64 = (f"data:image/jpeg;base64,{img_str}")
            images_b64_list.append(image_b64)
            # Save the image as PNG with unique timestamp
            filename = f"/workspace/sdxloutput/generated_image_{timestamp}_{i}.png"
            image.save(filename, format="PNG")

    return images_b64_list
    

last_prompt = ""
last_negative = ""
if os.path.isfile("last_options.txt"):
    fo = open("last_options.txt","r")
    options = fo.read().split("\n")
    if len(options) >= 2:
        last_prompt = options[0]
        last_negative = options[1]

while True:
    prompt = input("Prompt (use \"Q\" to quit)> ")
    if prompt.strip().lower() == "q":
        break
    if len(prompt) <= 0:
        if(len(last_prompt) <= 0):
            prompt = "A beautiful landscape"
        else:
            prompt = last_prompt
    last_prompt = prompt
    negative = input("Negative prompt (use \".\" for none)> ")
    if negative.strip() == ".":
        negative = ""
    elif negative.strip() == "":
        negative = last_negative
    last_negative = negative
    try:
        num_images = int(input("Number images> "))
        if num_images < 1:
            num_images = 1
    except:
        num_images = 1

    scale = 9
    samples = 1
    steps = 25
    refiner_strength = 0
    
    try:
        infer(prompt, negative, scale, samples, steps, refiner_strength, num_images)
    except Exception as e:
        eprint(str(e))

    fw = open("last_options.txt","w")
    fw.write(prompt+"\n")
    fw.write(negative+"\n")
    fw.close()

print("exiting")
