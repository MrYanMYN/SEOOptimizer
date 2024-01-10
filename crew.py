from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI

from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
def define_blog_topic(topic, supporting_tags, local=False):
    """

    :param topic: Current topic for article generation
    :param supporting_tags: related topic that would help to identify the subject
    :param local: Flag for determining if to use a local LLM
    :return: Returns {CrewAI} a crew object
    """
    if local:
        base_url = "http://localhost:1234/v1"
    else:
        base_url = "https://api.openai.com/v1"

    researcher = Agent(
      role='Senior Research Analyst',
      goal=f'Research and gain a deep understanding of {topic} which is probably based on {supporting_tags}',
      backstory="""You work at a leading consulting firm,
      Your expertise lies in identifying emerging trends and creating detailed explanation about certin topics.
      You have a knack for dissecting complex data and presenting
      actionable insights.""",
      verbose=True,
      allow_delegation=False,
      tools=[search_tool],
      # You can pass an optional llm attribute specifying what mode you wanna use.
      # It can be a local model through Ollama / LM Studio or a remote
      # model like OpenAI, Mistral, Antrophic of others (https://python.langchain.com/docs/integrations/llms/)
      #
      # Examples:
      # llm=ollama_llm # was defined above in the file
      llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7, base_url=base_url)
    )
    writer = Agent(
      role='Tech Content Strategist',
      goal=f'Craft compelling content on {topic}',
      backstory="""You are a renowned Content Strategist, known for
      your insightful and engaging articles.
      You transform complex concepts into compelling narratives.""",
      verbose=True,
      allow_delegation=True,
      # (optional) llm=ollama_llm
      llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7, base_url=base_url)
    )

    # Create tasks for your agents
    task1 = Task(
      description=f"""Conduct a comprehensive analysis of the {topic}.
      Identify key trends, breakthrough technologies, and potential industry impacts.
      Your final answer MUST be a full analysis report""",
      agent=researcher
    )

    task2 = Task(
      description=f"""Using the insights provided, develop an engaging blog
      post that highlights the most significant points in the subject: {topic}.
      Your post should be informative yet accessible, catering to a tech-savvy audience.
      Make it sound cool, avoid complex words so it doesn't sound like AI.
      Your final answer MUST be the full blog post of at least 4 paragraphs.""",
      agent=writer
    )

    # Instantiate your crew with a sequential process
    crew_composite = Crew(
      agents=[researcher, writer],
      tasks=[task1, task2],
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    return crew_composite