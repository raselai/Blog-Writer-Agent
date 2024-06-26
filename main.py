import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai_tools import YoutubeVideoSearchTool


os.environ['OPENAI_API_KEY'] = 'Your openai api key'

# Option 3: Pass the API key directly to the tool (less secure)
yt_tool = YoutubeVideoSearchTool(
    youtube_video_url='YouTube video url',
    openai_api_key='Your openai api key'
)
llm = ChatGroq(
    api_key="Your groq api key",
    model="llama3-8b-8192"
)

# YouTube video URL as a variable for easier modification
yt_tool = YoutubeVideoSearchTool(youtube_video_url="YouTube video url")

blog_researcher = Agent(
    role='Blog Researcher from YouTube Videos',
    goal="Get the relevant information from the YouTube video",
    verbose=True,
    memory=True,
    backstory=(
        'You work at a leading tech think tank.'
        'Your expertise lies in identifying emerging trends.'
        'You have a knack for dissecting complex data and presenting actionable insights. '
    ),
    tools=[yt_tool],
    allow_delegation=True,
    llm=llm
)

blog_writer = Agent(
    role='Blog Writer',
    goal="Narrate compelling stories about the YouTube video",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft "
        "engaging narratives that captivate and educate, bringing new "
        "discoveries to light in an accessible manner"
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm=llm
)

researcher_task = Task(
    description=(
        """Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts."""
    ),
    expected_output="Full analysis report in bullet points",
    tools=[yt_tool],
    agent=blog_researcher,
)

writer_task = Task(
    description=(
        "Using the insights provided, develop an engaging blog post that highlights "
        "human relationships with each other. Your post should be informative yet "
        "accessible, catering to an emotionally-savvy audience. Make it sound cool, "
        "avoid complex words so it doesn't sound like AI."
    ),
    expected_output="Full blog post of at least 4 paragraphs",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file="new-blog-post.md"
)

my_crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[researcher_task, writer_task],
    process=Process.sequential,
    full_output=True,
    verbose=True,
    memory=True,
    cache=True,
    max_rpm=10,  # Reduced from 100 to avoid potential rate limiting
    share_crew=True
)

try:
    result = my_crew.kickoff()
    print(result)
except Exception as e:
    print(f"An error occurred: {str(e)}")
