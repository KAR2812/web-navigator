#!/usr/bin/env python3
"""
main.py — CLI + voice input that creates the Agent and calls agent.print_response().
Drop into your project root and run: python main.py
"""
from src.nlu.intent_model import predict_intent

from dotenv import load_dotenv
import os
import sys
from typing import Optional

load_dotenv()

# voice helper (whisper). This version will return English-translated text
# if voice is in Hindi/Marathi when voice_to_text.listen_and_transcribe is configured.
try:
    from src.speech.voice_to_text import listen_and_transcribe
    _HAS_VOICE = True
except Exception:
    listen_and_transcribe = None
    _HAS_VOICE = False

# Try to import Agent, LLM, BrowserConfig (scripted agent)
_agent = None
_agent_available = False
try:
    from src.agent.web.browser.config import BrowserConfig
    from src.inference.gemini import ChatGemini
    from src.agent.web import Agent

    _agent_available = True
except Exception as e:
    print("Warning: Could not import Agent/LLM classes. Running in fallback mode.")
    print("Import error:", repr(e))
    _agent_available = False

def create_agent_from_env() -> Optional["Agent"]:
    """
    Create and return an Agent instance using environment variables.
    If creation fails, returns None.
    """
    if not _agent_available:
        return None
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        browser_instance_dir = os.getenv("BROWSER_INSTANCE_DIR")
        user_data_dir = os.getenv("USER_DATA_DIR")

        if not api_key:
            print("Warning: GOOGLE_API_KEY is not set in .env — LLM may fail.")

        llm = ChatGemini(model='gemini-2.5-flash-lite', api_key=api_key, temperature=0)
        config = BrowserConfig(
            browser=os.getenv("BROWSER", "edge"),
            browser_instance_dir=browser_instance_dir,
            user_data_dir=user_data_dir,
            headless=False,
        )

        agent_instance = Agent(config=config, llm=llm, verbose=True, use_vision=True, max_iteration=100)
        return agent_instance
    except Exception as e:
        print("Error creating agent from environment:", repr(e))
        return None

def get_user_query_via_cli() -> str:
    """
    Ask the user how they want to provide the query (type or voice).
    Returns the final user_query (string).
    """
    print("\nHow do you want to provide the query?")
    print("  1) Type the query")
    print("  2) Use voice (record from mic)")
    print("Press Enter at the first prompt to start voice recording as shortcut.")
    choice = input("Enter 1 or 2, or press Enter for voice: ").strip().lower()

    # Voice path (either blank Enter or explicit 2)
    if choice == "" or choice in ("2", "voice", "v"):
        if not _HAS_VOICE:
            print("Voice transcription not available. Please type your query.")
            return input("Type your query: ").strip()
        print("Starting voice recording (press Enter to stop). Speak in English/Hindi/Marathi.")
        # use tiny for speed; translate_after=True enables Marathi/Hindi->English translation
        text = listen_and_transcribe(model_name="tiny", verbose=True, force_language=None, translate_after=True)
        return text or ""

    # Typed path
    typed = input("Type your query and press Enter: ").strip()
    return typed

def main():
    global _agent

    # Try to create the scripted Agent
    if _agent_available:
        _agent = create_agent_from_env()
        if _agent is None:
            print("Agent creation failed — fallback mode active.")

    # Get user query (typed or voice → translated to English)
    user_query = get_user_query_via_cli()
    if not user_query:
        print("No query provided. Exiting.")
        sys.exit(0)

    print("\nUser said:", repr(user_query))

    # ✅ Run Intent Classifier
    intent, confidence = predict_intent(user_query)
    print(f"🧠 Detected Intent: {intent}  (confidence={confidence:.2f})")

    # ✅ Build an automation-ready command (natural language normalization)
    # This helps the agent understand the request more clearly
    intent_to_prompt = {
        "search_product": f"Search for {user_query}",
        "login": "Log into the account",
        "book_ticket": f"Book ticket: {user_query}",
        "open_dashboard": "Open the dashboard",
        "navigate": f"Navigate: {user_query}"
    }

    # fallback to original query if intent not in dictionary
    refined_query = intent_to_prompt.get(intent, user_query)

    print("🔍 Automation Prompt:", repr(refined_query))

    # ✅ Run agent automatically
    if _agent:
        try:
            print("🚀 Running Web Navigation Agent...")
            _agent.print_response(refined_query)
            return
        except Exception as e:
            print("⚠️ Agent Error:", repr(e))
            print("Falling back to printing query.")

    # Fallback (no agent available)
    print("USER QUERY (fallback):", refined_query)


if __name__ == "__main__":
    main()
