import os
import requests
import json
from typing import Generator, Dict, List, Any
from dotenv import load_dotenv
from .utils import ThinkingMarkers, StreamBuffer, get_model_thinking_markers
from .model_manager import model_manager

# Load environment variables early
load_dotenv()

def get_backend_llm_info() -> str:
    """Displays the LLM backend configuration."""
    current_model = model_manager.current_model
    if current_model and current_model.backend == "ollama":
        return f"--- Using Ollama Backend: {current_model.name} at {os.getenv('OLLAMA_HOST_URL', 'http://localhost:11434')} ---"
    return "Error: No valid model configuration found."

class OllamaClient:
    def __init__(self):
        self.host_url = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
        self.headers = {'Content-Type': 'application/json'}

    def _get_api_url(self) -> str:
        """Get the full API URL for chat endpoint."""
        return f"{self.host_url.rstrip('/')}/api/chat"

    def _create_payload(self, messages: List[Dict[str, str]]) -> Dict:
        current_model = model_manager.current_model
        if not current_model:
            raise ValueError("No model selected")
        
        return {
            "model": current_model.name,
            "messages": messages,
            "stream": True
        }

    def _handle_error(self, e: Exception) -> Generator[str, None, None]:
        api_url = self._get_api_url()
        if isinstance(e, requests.exceptions.ConnectionError):
            yield f"Error: Could not connect to Ollama at {api_url}. Is it running?"
        elif isinstance(e, requests.exceptions.RequestException):
            error_detail = str(e)
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_detail = error_json['error']
            except:
                pass
            yield f"Error communicating with Ollama: {error_detail}"
        else:
            print(f"Generic error during Ollama streaming call: {e}")
            yield f"An unexpected error occurred with Ollama: {str(e)}"

    def stream_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        current_model = model_manager.current_model
        if not current_model:
            yield "Error: No model selected"
            return

        # Get thinking markers for current model
        thinking_markers = get_model_thinking_markers(
            current_model.name,
            {
                "deepseek-r1": ("<think>", "</think>"),
                "qwen3": ("<think>", "</think>"),
            }
        )
        stream_buffer = StreamBuffer(thinking_markers)
        
        try:
            api_url = self._get_api_url()
            print(f"Sending to Ollama ({current_model.name}) with streaming...")
            with requests.post(
                api_url,
                headers=self.headers, 
                json=self._create_payload(messages), 
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        json_line = json.loads(line.decode('utf-8'))
                        
                        if 'message' in json_line and 'content' in json_line['message']:
                            content = json_line['message']['content']
                            yield from stream_buffer.process_chunk(content)
                        
                        if json_line.get('done', False):
                            yield from stream_buffer.flush()
                            break

        except Exception as e:
            yield from self._handle_error(e)

# Create a singleton instance
ollama_client = OllamaClient()

def get_llm_streaming_response(chat_history_messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Gets a streaming response from the configured LLM backend.
    Args:
        chat_history_messages: List of dictionaries in OpenAI message format
        e.g., [{"role": "user", "content": "Hi"}, ...]
    Yields:
        Chunks of the LLM's response content as they arrive.
    """
    current_model = model_manager.current_model
    if not current_model:
        yield "Error: No model selected. Please select a model from the Model Selection menu."
        return
        
    if current_model.backend == "ollama":
        yield from ollama_client.stream_response(chat_history_messages)
    else:
        print(f"Error: Invalid model backend specified: {current_model.backend}")
        yield "Error: Invalid model backend configuration. Please check server logs."