# src/llm_client.py
import os
import openai
import requests
import json
from typing import Generator, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# --- Configuration ---
LLM_BACKEND = os.getenv("LLM_BACKEND") # Should be 'openai' or 'ollama'

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama Config
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434") # Default Ollama host
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama:latest") # Required if using Ollama
# Accepted values: "tinyllama:latest" or "gemma3:1b"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") # Required if using Ollama

def get_backend_llm_info() -> str:
    """Displays the LLM backend configuration."""
    info = f"LLM_BACKEND: {LLM_BACKEND}\n"
    if LLM_BACKEND == "openai":
        info += f"--- Using OpenAI Backend: {OPENAI_MODEL} ---"
    elif LLM_BACKEND == "ollama":
        info += f"--- Using Ollama Backend: {OLLAMA_MODEL} at {OLLAMA_HOST_URL} ---"
    else:
        info += f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'openai' or 'ollama'."
    return info

# --- OpenAI Streaming Client ---
def _get_openai_streaming_response(history_messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Gets streaming response from OpenAI API."""
    if not OPENAI_API_KEY:
        yield "Error: OPENAI_API_KEY not configured."
        return

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print(f"Sending to OpenAI ({OPENAI_MODEL}) with streaming...")
        
        # Create a streaming response
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history_messages,
            temperature=0.7,
            stream=True,  # Enable streaming
        )
        
        # Collect chunks and yield them
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
                
    except openai.AuthenticationError:
        yield "Error: OpenAI authentication failed. Check API key."
    except openai.RateLimitError:
        yield "Error: OpenAI rate limit exceeded."
    except Exception as e:
        print(f"Error during OpenAI streaming call: {e}")
        yield f"Error communicating with OpenAI: {str(e)}"

# --- Ollama Streaming Client ---
def _get_ollama_streaming_response(history_messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Gets streaming response from a local Ollama instance."""
    if not OLLAMA_MODEL:
        yield "Error: OLLAMA_MODEL environment variable not set."
        return

    ollama_api_url = f"{OLLAMA_HOST_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": history_messages,
        "stream": True  # Enable streaming
    }
    headers = {'Content-Type': 'application/json'}

    try:
        print(f"Sending to Ollama ({OLLAMA_MODEL} at {OLLAMA_HOST_URL}) with streaming...")
        
        # Send request and process the streaming response
        with requests.post(ollama_api_url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # Process each line in the streaming response
            for line in response.iter_lines():
                if line:
                    # Decode the line and parse as JSON
                    json_line = json.loads(line.decode('utf-8'))
                    
                    # Extract content from the streaming response
                    if 'message' in json_line and 'content' in json_line['message']:
                        content = json_line['message']['content']
                        yield content
                    
                    # Check if this is the final message
                    if json_line.get('done', False):
                        break

    except requests.exceptions.ConnectionError:
        yield f"Error: Could not connect to Ollama at {ollama_api_url}. Is it running?"
    except requests.exceptions.RequestException as e:
        print(f"Error during Ollama streaming call: {e}")
        error_detail = str(e)
        try:
            error_json = e.response.json()
            if 'error' in error_json:
                error_detail = error_json['error']
        except:
            pass
        yield f"Error communicating with Ollama: {error_detail}"
    except Exception as e:
        print(f"Generic error during Ollama streaming call: {e}")
        yield f"An unexpected error occurred with Ollama: {str(e)}"

# --- Non-streaming fallback methods ---
def _get_openai_response(history_messages: list[dict]) -> str:
    """Gets response from OpenAI API (non-streaming)."""
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

def _get_ollama_response(history_messages: list[dict]) -> str:
    """Gets response from a local Ollama instance (non-streaming)."""
    if not OLLAMA_MODEL:
        return "Error: OLLAMA_MODEL environment variable not set."

    ollama_api_url = f"{OLLAMA_HOST_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": history_messages,
        "stream": False  # Get the full response at once
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

# --- Main Dispatcher Functions ---
def get_llm_response(chat_history_messages: list[dict]) -> str:
    """
    Gets a response from the configured LLM backend (non-streaming).

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

def get_llm_streaming_response(chat_history_messages: list[dict]) -> Generator[str, None, None]:
    """
    Gets a streaming response from the configured LLM backend.

    Args:
        chat_history_messages: List of dictionaries in OpenAI message format
                                e.g., [{"role": "user", "content": "Hi"}, ...]
    Yields:
        Chunks of the LLM's response content as they arrive.
    """
    if LLM_BACKEND == "openai":
        yield from _get_openai_streaming_response(chat_history_messages)
    elif LLM_BACKEND == "ollama":
        yield from _get_ollama_streaming_response(chat_history_messages)
    else:
        print(f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'openai' or 'ollama'.")
        yield "Error: LLM backend misconfigured. Please check server logs/environment variables."