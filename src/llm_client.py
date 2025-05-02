# src/llm_client.py
import os
import requests
import json
from typing import Generator, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# --- Configuration ---
LLM_BACKEND = os.getenv("LLM_BACKEND") # Should be 'openai' or 'ollama'

# Ollama Config
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434") # Default Ollama host
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") # Ollama model name. For the latest list: https://ollama.com/search

def get_backend_llm_info() -> str:
    """Displays the LLM backend configuration."""
    info = f"LLM_BACKEND: {LLM_BACKEND}\n"
    if LLM_BACKEND == "ollama":
        info += f"--- Using Ollama Backend: {OLLAMA_MODEL} at {OLLAMA_HOST_URL} ---"
    else:
        info += f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'ollama'."
    return info


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


def get_llm_streaming_response(chat_history_messages: list[dict]) -> Generator[str, None, None]:
    """
    Gets a streaming response from the configured LLM backend.

    Args:
        chat_history_messages: List of dictionaries in OpenAI message format
                                e.g., [{"role": "user", "content": "Hi"}, ...]
    Yields:
        Chunks of the LLM's response content as they arrive.
    """
    if LLM_BACKEND == "ollama":
        yield from _get_ollama_streaming_response(chat_history_messages)
    else:
        print(f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'ollama'.")
        yield "Error: LLM backend misconfigured. Please check server logs/environment variables."