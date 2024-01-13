import base64
from openai import OpenAI
from langchain.tools import tool
import deepdanbooru as dd
import numpy as np
import PIL.Image
import tensorflow as tf
import huggingface_hub
from clip_interrogator import Config, Interrogator
import dotenv

"""
Base Classes
"""
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool


class GPTVInput(BaseModel):
    prompt: str = Field(description="a prompt that asks a question about the given image")
    image_path: str = Field(description="local image path")


class DeepDanInput(BaseModel):
    image_path: str = Field(description="local image path")


class ClipInterrogatorInput(BaseModel):
    image_path: str = Field(description="local image path")

class VisionTools:
    def __init__(self, is_deepdan):
        dotenv.load_dotenv()
        if is_deepdan:
            self.dd_model = self.load_deepdan_model()
            self.dd_labels = self.load_deepdan_labels()

    @staticmethod
    def load_deepdan_model() -> tf.keras.Model:
        path = huggingface_hub.hf_hub_download('public-data/DeepDanbooru',
                                               'model-resnet_custom_v3.h5')
        model = tf.keras.models.load_model(path)
        return model

    @staticmethod
    def load_deepdan_labels() -> list[str]:
        path = huggingface_hub.hf_hub_download('public-data/DeepDanbooru',
                                               'tags.txt')
        with open(path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_file.close()
        return encoded_image

    @tool('image_question_answerer', args_schema=DeepDanInput, return_direct=True)
    def gpt_v_predict(self, prompt: str, image_path: str) -> str:
        """Given a prompt and an image , the model returns an answer about a visual aspect of the image."""
        client = OpenAI()
        base64_image = self.encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        answer = response.choices[0].message.content
        return answer

    @tool('image_contents_describer', args_schema=ClipInterrogatorInput, return_direct=True)
    def clip_interrogator_predict(self, image_path: str) -> str:
        """
        Runs the clip model to receive a description of what is seen in the given image.
        """
        image = PIL.Image.open(image_path).convert('RGB')
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        return ci.interrogate(image)

    @tool('image_tag_analyzer', args_schema=DeepDanInput, return_direct=True)
    def deepdan_predict(
            self, image_path: str, score_threshold: float = 0.8
    ) -> str:
        """Given a prompt and an image , the model returns an answer about a visual aspect of the image.
        """
        _, height, width, _ = self.dd_model.input_shape
        image = PIL.Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        image = tf.image.resize(image,
                                size=(height, width),
                                method=tf.image.ResizeMethod.AREA,
                                preserve_aspect_ratio=True)
        image = image.numpy()
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.
        probs = self.dd_model.predict(image[None, ...])[0]
        probs = probs.astype(float)

        indices = np.argsort(probs)[::-1]
        result_all = dict()
        result_threshold = dict()
        for index in indices:
            label = self.dd_labels[index]
            prob = probs[index]
            result_all[label] = prob
            if prob < score_threshold:
                break
            result_threshold[label] = prob
        result_text = ', '.join(result_all.keys())
        return result_text
