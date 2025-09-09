import runpod
import base64


def handler(job):
    input_data = job["input"]
    prompt = input_data["prompt"]
    image_base64 = input_data["image"]
    # Base 64 decode the image
    image_data = None
    if image_base64:
        image_data = base64.b64decode(image_base64)

    return {
        "prompt": prompt,
        "image_data_length": len(image_data) if image_data else 0
    }


runpod.serverless.start({"handler": handler})
