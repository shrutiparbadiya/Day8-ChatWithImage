from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    DetrImageProcessor,
    DetrForObjectDetection
)

# Dummy schema (required by BaseTool)
class ImageInput(BaseModel):
    image_path: str

class ImageCaptionTool(BaseTool):
    name: str = "Image Captioner"
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )
    args_schema: Type[BaseModel] = ImageInput

    def _run(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name: str = "Object Detector"
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )
    args_schema: Type[BaseModel] = ImageInput

    def _run(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += f'[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] '
            detections += f'{model.config.id2label[int(label)]} {float(score)}\n'

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
