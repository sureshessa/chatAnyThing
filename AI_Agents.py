from crewai_tools import tool
import requests
import os

@tool
def search_google(query: str) -> str:
    """
    Perform a Google search using SerpAPI and return the results.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: Serper API key is not set."
    
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"
    
# Create the agent
researcher = Agent(
    role='Researcher',
    goal='Find information on {topic} using Llama3 and fallback on SerpAPI when LLM has No Answer!',
    llm=llm,#default gpt-4.0
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert researcher, skilled at using both advanced AI models and online resources to find accurate information quickly."
    ),
    tools=[search_google]
)
# Define a task for the agent
research_task = Task(
    description=(
        "Use the Llama3 model to find information on {topic}. If the model cannot provide a sufficient answer, perform a Google search."
        "Your final answer must include the top 3 results with summaries."
    ),
    expected_output='A list of top 3 search results with summaries.',
    tools=[search_google],
    agent=researcher,
)

# Form a crew with the agent and the task
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

# Kickoff the crew with a specific topic
result = crew.kickoff(inputs={'topic': 'Give me latest info of AI in the 2024'})
print(result)