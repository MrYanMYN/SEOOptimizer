import base64

from openai import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

search_tool = DuckDuckGoSearchRun()


def image_tools(prompt, image_path):
    client = OpenAI()
    base64_image = encode_image(image_path)

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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_file.close()
    return encoded_image
