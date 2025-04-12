# src/llm_client.py
import os
import openai
import requests
import json
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# --- Configuration ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower() # Default to openai

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama Config
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434") # Default Ollama host
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama:latest") # Required if using Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b") # Required if using Ollama


# --- OpenAI Client ---
def _get_openai_response(history_messages: list[dict]) -> str:
    """Gets response from OpenAI API."""
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not configured."
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY) # Explicit client creation
        print(f"Sending to OpenAI ({OPENAI_MODEL})...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history_messages,
            temperature=0.7,
        )
        bot_message = response.choices[0].message.content
        return bot_message.strip()
    except openai.AuthenticationError:
         return "Error: OpenAI authentication failed. Check API key."
    except openai.RateLimitError:
         return "Error: OpenAI rate limit exceeded."
    except Exception as e:
        print(f"Error during OpenAI call: {e}")
        return f"Error communicating with OpenAI: {str(e)}"

# --- Ollama Client ---
def _get_ollama_response(history_messages: list[dict]) -> str:
    """Gets response from a local Ollama instance."""
    if not OLLAMA_MODEL:
        return "Error: OLLAMA_MODEL environment variable not set."

    ollama_api_url = f"{OLLAMA_HOST_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": history_messages,
        "stream": False # Get the full response at once
    }
    headers = {'Content-Type': 'application/json'}

    try:
        print(f"Sending to Ollama ({OLLAMA_MODEL} at {OLLAMA_HOST_URL})...")
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        # Ensure the expected keys exist
        if "message" in response_data and "content" in response_data["message"]:
            return response_data["message"]["content"].strip()
        else:
             print(f"Unexpected Ollama response format: {response_data}")
             return "Error: Unexpected response format from Ollama."

    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Ollama at {ollama_api_url}. Is it running?"
    except requests.exceptions.RequestException as e:
        print(f"Error during Ollama call: {e}")
        # Try to get more specific error from response body if available
        error_detail = str(e)
        try:
            error_json = e.response.json()
            if 'error' in error_json:
                error_detail = error_json['error']
        except: # Ignore parsing errors
            pass
        return f"Error communicating with Ollama: {error_detail}"
    except Exception as e:
        print(f"Generic error during Ollama call: {e}")
        return f"An unexpected error occurred with Ollama: {str(e)}"


# --- Main Dispatcher Function ---
def get_llm_response(chat_history_messages: list[dict]) -> str:
    """
    Gets a response from the configured LLM backend (OpenAI or Ollama).

    Args:
        chat_history_messages: List of dictionaries in OpenAI message format
                                e.g., [{"role": "user", "content": "Hi"}, ...]
    Returns:
        The LLM's response content as a string, or an error message.
    """
    if LLM_BACKEND == "openai":
        return _get_openai_response(chat_history_messages)
    elif LLM_BACKEND == "ollama":
        return _get_ollama_response(chat_history_messages)
    else:
        print(f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'openai' or 'ollama'.")
        return "Error: LLM backend misconfigured. Please check server logs/environment variables."