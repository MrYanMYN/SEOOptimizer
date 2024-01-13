
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import tools.vision_tools as vt

search_tool = DuckDuckGoSearchRun()
vision_tools = vt.VisionTools(is_deepdan=True)