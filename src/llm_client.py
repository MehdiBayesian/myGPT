import os
import requests
import json
import re
from typing import Generator, Dict, List, Any
from dotenv import load_dotenv
# Load environment variables early
load_dotenv()
# --- Configuration ---
LLM_BACKEND = os.getenv("LLM_BACKEND") # Should be 'openai' or 'ollama'
# Ollama Config
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434") # Default Ollama host
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") # Ollama model name. For the latest list: https://ollama.com/search
# Adding this to extract parent model name, e.g. deepseek-r1:4b and deepseek-r1:8b have same base name (deepseek-r1)
OLLAMA_MODEL_BASE = OLLAMA_MODEL.split(':')[0] if ':' in OLLAMA_MODEL else OLLAMA_MODEL
print(OLLAMA_MODEL_BASE)

# --- Thinking markers configuration ---
# Dictionary mapping model names to their thinking markers
# TODO: I need to fix this to be automatic based on model params. Seems like ollma has model params dictionary card that could be used? 
THINKING_MARKERS = {
    "deepseek-coder": ("<think>", "</think>"),
    "deepseek-r1": ("<think>", "</think>"),
    # Add more models and their thinking markers as needed
}
# START_THINKING_MESSAGE = "🤔 [Started Thinking ...] "
# END_THINKING_MESSAGE = " [... Done Thinking] 💡"
START_THINKING_MESSAGE = "🤔 [Started Tak-Navazi ...] "
END_THINKING_MESSAGE = " [... Done Tak-Navazi] 🏁 "


def get_backend_llm_info() -> str:
    """Displays the LLM backend configuration."""
    info = f"LLM_BACKEND: {LLM_BACKEND}\n"
    if LLM_BACKEND == "ollama":
        info += f"--- Using Ollama Backend: {OLLAMA_MODEL} at {OLLAMA_HOST_URL} ---"
    else:
        info += f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'ollama'."
    return info

# --- Ollama Streaming Client ---
def get_ollama_streaming_response(history_messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Gets streaming response from a local Ollama instance with thinking marker detection."""
    if not OLLAMA_MODEL:
        yield "Error: OLLAMA_MODEL environment variable not set."
        return
    
    # Get the thinking markers for the current model (if available)
    start_thinking, end_thinking = THINKING_MARKERS.get(
        OLLAMA_MODEL_BASE.lower(), 
        (None, None)  # Default to None if model doesn't have thinking markers
    )
    
    # Buffer to hold partial content across multiple streaming chunks
    buffer = ""
    in_thinking_mode = False
    
    ollama_api_url = f"{OLLAMA_HOST_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": history_messages,
        "stream": True # Enable streaming
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
                        
                        # If the model has thinking markers, process them
                        if start_thinking and end_thinking:
                            # Append the new content to our buffer
                            buffer += content
                            
                            # Check for start thinking marker
                            if start_thinking in buffer and not in_thinking_mode:
                                # Split by the start marker to get content before and after
                                parts = buffer.split(start_thinking, 1)
                                if len(parts) > 1:
                                    # Yield content before the marker
                                    if parts[0]:
                                        yield parts[0]
                                    # Yield our replacement for start thinking
                                    yield START_THINKING_MESSAGE
                                    # Reset buffer to content after marker
                                    buffer = parts[1]
                                    in_thinking_mode = True
                            
                            # Check for end thinking marker
                            if end_thinking in buffer and in_thinking_mode:
                                # Split by the end marker
                                parts = buffer.split(end_thinking, 1)
                                if len(parts) > 1:
                                    # Yield content before the marker (which is part of thinking)
                                    if parts[0]:
                                        yield parts[0]
                                    # Yield our replacement for end thinking
                                    yield END_THINKING_MESSAGE
                                    # Reset buffer to content after marker
                                    buffer = parts[1]
                                    in_thinking_mode = False
                            
                            # If no markers found in this chunk, or after processing markers,
                            # yield any remaining complete content in the buffer
                            if not any(marker in buffer for marker in [start_thinking, end_thinking]):
                                yield buffer
                                buffer = ""
                        else:
                            # If no thinking markers configured for this model, just yield the content
                            yield content
                    
                    # Check if this is the final message
                    if json_line.get('done', False):
                        # If there's any remaining content in the buffer, yield it
                        if buffer:
                            yield buffer
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
        yield from get_ollama_streaming_response(chat_history_messages)
    else:
        print(f"Error: Invalid LLM_BACKEND specified: {LLM_BACKEND}. Use 'ollama'.")
        yield "Error: LLM backend misconfigured. Please check server logs/environment variables."