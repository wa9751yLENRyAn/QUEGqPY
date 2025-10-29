import requests
import torch
from PIL import Image
from prismatic import load

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
vlm = load("Open-Qwen2VL")
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = [vlm.vision_backbone.image_transform(Image.open(requests.get(image_url, stream=True).raw).convert("RGB")).unsqueeze(0)]
user_prompt = '<image>' + '\n' + "Describe the image."

# Generate!
generated_text = vlm.generate_batch(
    image,
    [user_prompt],
    do_sample=False,
    max_new_tokens=512,
    min_length=1,
)
print(generated_text[0])