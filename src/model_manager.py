from typing import List, Tuple, Optional
import os
import subprocess
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

@dataclass
class ModelConfig:
    name: str  # Full model name (e.g., "deepseek-r1:4b")
    display_name: str  # Display name in UI (e.g., "DeepSeek R1 4B")
    backend: str  # e.g., "ollama"

class ModelManager:
    def __init__(self):
        self._current_model: Optional[ModelConfig] = None
        self._available_models: List[ModelConfig] = []
        self._load_available_models()

    def _get_ollama_models(self) -> List[str]:
        """Get list of installed Ollama models."""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                # Skip header line and parse model names
                lines = result.stdout.strip().split('\n')[1:]
                models = []
                for line in lines:
                    if line.strip():
                        # First column is the model name
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except Exception as e:
            print(f"Error getting Ollama models: {e}")
            return []

    def _format_display_name(self, model_name: str) -> str:
        """Convert model name to display name."""
        # Remove 'latest' tag if present
        if model_name.endswith(':latest'):
            model_name = model_name[:-7]
        
        # Split name and version
        parts = model_name.split(':')
        base_name = parts[0]
        version = parts[1] if len(parts) > 1 else ''
        
        # Capitalize words
        words = base_name.replace('-', ' ').split()
        capitalized = ' '.join(word.capitalize() for word in words)
        
        # Add version if present
        if version:
            return f"{capitalized} ({version})"
        return capitalized

    def _load_available_models(self):
        """Load available models from Ollama and set default from environment."""
        # Get available models
        ollama_models = self._get_ollama_models()
        self._available_models = [
            ModelConfig(
                name=model_name,
                display_name=self._format_display_name(model_name),
                backend="ollama"
            )
            for model_name in ollama_models
        ]
        
        # Get default model from environment
        env_model = os.getenv("OLLAMA_MODEL")
        
        if env_model:
            print(f"Looking for default model from .env: {env_model}")
            # Try to find the exact model from environment
            for model in self._available_models:
                if model.name == env_model:
                    self._current_model = model
                    print(f"Found and set default model: {model.name}")
                    break
            
            if not self._current_model:
                print(f"Warning: Model {env_model} specified in .env not found in available models")
        
        # If no model is set (either no env var or model not found), use first available
        if not self._current_model and self._available_models:
            self._current_model = self._available_models[0]
            print(f"Using first available model as default: {self._current_model.name}")
            # Update environment variable to match
            os.environ["OLLAMA_MODEL"] = self._current_model.name

    @property
    def current_model(self) -> Optional[ModelConfig]:
        return self._current_model

    @current_model.setter
    def current_model(self, model_name: str):
        """Set current model by name."""
        for model in self._available_models:
            if model.name == model_name:
                self._current_model = model
                # Update environment variable
                os.environ["OLLAMA_MODEL"] = model_name
                print(f"Model changed to: {model_name}")
                break

    def get_model_choices(self) -> List[Tuple[str, str]]:
        """Get model choices in Gradio-friendly format (display_name, name)."""
        return [(model.display_name, model.name) for model in self._available_models]

    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get model config by name."""
        for model in self._available_models:
            if model.name == name:
                return model
        return None

# Create a singleton instance
model_manager = ModelManager() 