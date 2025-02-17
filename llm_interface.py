# llm_interface.py
from typing import Optional
from langchain_ollama import OllamaLLM  # Updated import
from config import Config
from utils import logger
from config import load_config

class LLMInterface:
    def __init__(self, config: Config):
        self.config = config
        self.primary_model_name = config.llm.model_name
        self.fallback_model_name = config.llm.fallback_model_name
        self.model_params = config.llm.params
        self.llm = None  # Initialize in _load_model

    def _load_model(self, model_name):
        try:
            logger.info(f"Loading Ollama model: {model_name}")
            # Initialize with base parameters
            model = OllamaLLM(
                model=model_name,
                temperature=self.model_params.get('temperature', 0.7),
                top_p=self.model_params.get('top_p', 0.9),
                streaming=True
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def initialize_llm(self):
        self.llm = self._load_model(self.primary_model_name)
        if self.llm is None:
            logger.warning(f"Failed to load primary model {self.primary_model_name}. Attempting fallback model {self.fallback_model_name}")
            self.llm = self._load_model(self.fallback_model_name)
            if self.llm is None:
                raise Exception("Failed to load both primary and fallback LLM models. Please ensure Ollama is running and models are available.")
            else:
                logger.info(f"Fallback model {self.fallback_model_name} loaded successfully.")
        else:
            logger.info(f"Primary model {self.primary_model_name} loaded successfully.")
        return self.llm

    def generate_text(self, prompt: str, model_name: Optional[str] = None, params_override: Optional[dict] = None) -> str:
        """
        Generates text using the LLM. Allows overriding model and parameters for specific tasks.
        """
        llm_to_use = self.llm  # Default to initialized LLM
        if model_name:
            llm_to_use = self._load_model(model_name)
            if not llm_to_use:
                logger.warning(f"Requested model {model_name} failed to load. Using default LLM.")
                llm_to_use = self.llm  # Revert to default if requested model fails

        if not llm_to_use:
            raise Exception("No LLM available for text generation.")

        # Merge default params with overrides
        params = { } 
        if params_override:
            params.update(params_override)

        try:
            response = ""
            # Use the stream method with the updated parameters
            for chunk in llm_to_use.stream(prompt, **params):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                print(chunk_text, end="", flush=True)
                response += chunk_text
            return response
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return ""  # Or raise exception depending on error handling strategy


# Example usage (for testing):
if __name__ == "__main__":
    config = load_config()
    llm_interface = LLMInterface(config)
    llm_interface.initialize_llm()
    prompt_test = "Write a short summary about the impact AI in developing nations in Africa."
    generated_text = llm_interface.generate_text(prompt_test)
    print("\nGenerated Text:\n", generated_text)