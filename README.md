<div align="center">

  <h1>🌐 Web-Navigator</h1>

  <a href="https://github.com/Jeomon/Web-Agent/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Powered%20by-Playwright-45ba63?logo=playwright&logoColor=white" alt="Powered by Playwright">
  <br>

  <a href="https://x.com/CursorTouch">
    <img src="https://img.shields.io/badge/follow-%40CursorTouch-1DA1F2?logo=twitter&style=flat" alt="Follow on Twitter">
  </a>
  <a href="https://discord.com/invite/Aue9Yj2VzS">
    <img src="https://img.shields.io/badge/Join%20on-Discord-5865F2?logo=discord&logoColor=white&style=flat" alt="Join us on Discord">
  </a>

</div>

<br>

**Web Navigator** is your intelligent browsing companion, built to seamlessly navigate websites, interact with dynamic content, perform smart searches, download files, and adapt to ever-changing pages — all with minimal effort from you. Powered by advanced LLMs and the robust Playwright framework, it transforms complex web tasks into streamlined, automated workflows that boost productivity and save time.

## 🛠️Installation Guide

### **Prerequisites**

- Python 3.11 or higher
- UV

### **Installation Steps**

**Clone the repository:**

```bash
git clone https://github.com/CursorTouch/Web-Navigator.git
cd Web-Navigator
```

**Install dependencies:**

```bash
uv sync
```

**Setup Playwright:**

```bash
playwright install
```

---

**Setting up the `.env` file:**

```bash
GOOGLE_API_KEY=""
```

Basic setup of the agent.

```python
from src.inference.gemini import ChatGemini
from src.agent.web import WebAgent
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key=os.getenv('GOOGLE_API_KEY')

llm=ChatGemini(model='gemini-2.0-flash',api_key=google_api_key,temperature=0)
agent=Agent(llm=llm,verbose=True,use_vision=False)

user_query=input('Enter your query: ')
agent_response=agent.invoke(user_query)
print(agent_response.get('output'))

```

Execute the following command to start the agent:

```bash
python app.py
```




- **[Playwright Documentation](https://playwright.dev/docs/intro)**  
- **[LangGraph Examples](https://github.com/langchain-ai/langgraph/blob/main/examples/web-navigation/web_voyager.ipynb)**  
- **[vimGPT](https://github.com/ishan0102/vimGPT)**  
- **[WebVoyager](https://github.com/MinorJerry/WebVoyager)**  
