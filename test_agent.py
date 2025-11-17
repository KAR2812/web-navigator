from src.inference.gemini import ChatGemini
from src.agent.web import Agent
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize LLM (Gemini)
llm = ChatGemini(model='gemini-2.0-flash', api_key=google_api_key, temperature=0)

# Initialize Agent
agent = Agent(llm=llm, verbose=True, use_vision=False)

# Take user input
user_query = input('Enter your query: ')

# Run agent
agent_response = agent.invoke(user_query)
print(agent_response.get('output'))
