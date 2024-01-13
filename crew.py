from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from tools.tools import search_tool, vision_tools

# Define your agents with roles and goals
def define_pipeline(topic, supporting_tags, local=False):
    """
    :param topic: Current topic for article generation
    :param supporting_tags: related topic that would help to identify the subject
    :param local: Flag for determining if to use a local LLM
    :return: Returns {CrewAI} a crew object
    """

    # Create agents
    researcher, seoExpert, writer = Agents(supporting_tags, topic, local).get_agents()

    # Create tasks for your agents
    tasks = Tasks([researcher, seoExpert, writer], topic)

    # Instantiate your crew with a sequential process
    crew_composite = Crew(
      agents=[researcher, seoExpert, writer],
      tasks=tasks.get_tasks(),
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    return crew_composite


class Tasks():
    def __init__(self, agents, topic: str):
        self.agents = agents
        self.topic = topic
        self.tasks = []
        self.create_tasks()

    def get_tasks(self):
        return self.tasks

    def create_tasks(self):
        task1 = Task(
            description=f"""Conduct a comprehensive analysis of the {self.topic}.
          Identify key trends, breakthrough technologies, and potential industry impacts.
          Your final answer MUST be a full analysis report""",
            agent=self.agents[0]#researcher
        )
        task2 = Task(
            description=f"""Conduct a comprehensive analysis of the {self.topic}.
              Identify key trends and most searched questions.
              Your final answer MUST be a list of questions the most asked questions on the subject: {self.topic}""",
            agent=self.agents[1]#seoExpert
        )
        task3 = Task(
            description=f"""Using the insights and the list of questions provided, develop an engaging blog
          post that highlights the most significant points in the subject: {self.topic}.
          Your post should be informative yet accessible.
          Make it sound cool, avoid complex words so it doesn't sound like AI.
          Your final answer MUST be the full blog post of at least 4 paragraphs and must answer all of the asked questions in a Q&A format.
        """,
            agent=self.agents[2]#writer
        )

        self.tasks = [task1, task2, task3]


class Agents():
    def __init__(self, supporting_tags, topic, local):
        self.topic = topic
        self.supporting_tags = supporting_tags
        if local:
            self.base_url = "http://localhost:1234/v1"
        else:
            self.base_url = "https://api.openai.com/v1"

    def get_agents(self):
        return self.researcher(), self.seo_expert(), self.writer()

    def seo_expert(self):
        return Agent(
        role='SEO Expert',
        goal=f'Research keywords and questions to optimize SEO for {self.topic} which is probably based on {self.supporting_tags}',
        backstory="""You work at a leading SEO Consulting firm,
        Your expertise lies in optimizing SEO of given subject and increasing their visibility.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, vision_tools.gpt_v_predict],
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, base_url=self.base_url)
    )

    def researcher(self):
        return Agent(
        role='Senior Research Analyst',
        goal=f'Research and gain a deep understanding of {self.topic} which is probably based on {self.supporting_tags}',
        backstory="""You work at a leading consulting firm,
      Your expertise lies in identifying emerging trends and creating detailed explanation about certin topics.
      You have a knack for dissecting complex data and presenting
      actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, vision_tools.gpt_v_predict],
        # You can pass an optional llm attribute specifying what mode you wanna use.
        # It can be a local model through Ollama / LM Studio or a remote
        # model like OpenAI, Mistral, Antrophic of others (https://python.langchain.com/docs/integrations/llms/)
        #
        # Examples:
        # llm=ollama_llm # was defined above in the file
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, base_url=self.base_url)
    )

    def writer(self):
        return Agent(
        role='Content Strategist',
        goal=f'Craft compelling content on {self.topic}',
        backstory="""You are a renowned Content Strategist, known for
      your insightful and engaging articles.
      You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True,
        # (optional) llm=ollama_llm
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, base_url=self.base_url)
    )
