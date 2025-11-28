"""
LLM Client with support for multiple providers.
Supports HuggingFace Inference API, OpenAI, and local models.
"""
import os
import logging
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Inference API client with robust URL handling."""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize HuggingFace client.
        
        Args:
            api_token: HuggingFace API token
        """
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        
        # Models to try in order
        self.models = [
            "google/flan-t5-large",  # Often most reliable free model
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        
        # URL patterns to try
        self.url_patterns = [
            "https://api-inference.huggingface.co/models/{model}",
            "https://router.huggingface.co/models/{model}",
            "https://router.huggingface.co/hf-inference/models/{model}"
        ]
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate response trying multiple models and URLs."""
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        last_error = None
        
        for model in self.models:
            for pattern in self.url_patterns:
                url = pattern.format(model=model)
                try:
                    # logger.info(f"Trying model {model} at {url}")
                    response = requests.post(url, headers=headers, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "")
                        elif isinstance(result, dict) and "generated_text" in result:
                            return result["generated_text"]
                        else:
                            return str(result)
                    elif response.status_code == 403:
                        logger.warning(f"403 Forbidden for {url}. Check token permissions.")
                        last_error = Exception("Token permission denied (403). Enable 'Inference' in HF token settings.")
                    else:
                        # logger.warning(f"Failed {url}: {response.status_code}")
                        last_error = Exception(f"HTTP {response.status_code}: {response.text}")
                        
                except Exception as e:
                    # logger.warning(f"Exception for {url}: {e}")
                    last_error = e
                    
        # If we get here, all attempts failed
        logger.error(f"All LLM attempts failed. Last error: {last_error}")
        raise last_error

    def is_available(self) -> bool:
        """Check if HuggingFace API is available."""
        return bool(self.api_token)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)


class FallbackLLMClient(BaseLLMClient):
    """Fallback client that uses a simple response when no LLM is available."""
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate a fallback response."""
        return (
            "I apologize, but I'm currently unable to generate a response as no LLM provider is configured. "
            "Please set up either HuggingFace API token (HUGGINGFACE_API_TOKEN) or OpenAI API key (OPENAI_API_KEY) "
            "in your environment variables or .env file."
        )
    
    def is_available(self) -> bool:
        """Fallback is always available."""
        return True


class LocalLLMClient(BaseLLMClient):
    """Local LLM client using ctransformers."""
    
    def __init__(self, model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        """
        Initialize local client.
        
        Args:
            model_path: Path to the GGUF model file
        """
        self.model_path = model_path
        self.llm = None
        self._load_model()
        
    def _load_model(self):
        """Load the model if it exists."""
        if os.path.exists(self.model_path):
            try:
                from ctransformers import AutoModelForCausalLM
                logger.info(f"Loading local model from {self.model_path}...")
                # Try loading with GPU offloading first
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        os.path.abspath(self.model_path),
                        model_type="llama",
                        gpu_layers=50,
                        context_length=2048
                    )
                    logger.info("Local model loaded successfully with GPU support!")
                except Exception as gpu_error:
                    logger.warning(f"Failed to load with GPU: {gpu_error}. Falling back to CPU...")
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        os.path.abspath(self.model_path),
                        model_type="llama",
                        gpu_layers=0,
                        context_length=2048
                    )
                    logger.info("Local model loaded successfully on CPU!")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.llm = None
        else:
            logger.warning(f"Local model not found at {self.model_path}")
            
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate response using local model."""
        if not self.llm:
            raise Exception("Local model not loaded")
            
        try:
            # ctransformers generate returns a generator or text
            response = self.llm(
                prompt, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,  # Penalize repetition to prevent loops
                stop=["</s>", "<|user|>", "<|assistant|>", "USER QUESTION:"] # Stop generation at these tokens
            )
            return response
        except Exception as e:
            logger.error(f"Local inference error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if local model is loaded."""
        return self.llm is not None


class MedicalLLMClient:
    """Medical chatbot LLM client with prompt engineering."""
    
    def __init__(self):
        """Initialize the medical LLM client with provider selection."""
        self.client = self._select_provider()
        logger.info(f"Using LLM provider: {type(self.client).__name__}")
    
    def _select_provider(self) -> BaseLLMClient:
        """Select the best available LLM provider."""
        # 1. Try Local LLM first if configured or available
        local_client = LocalLLMClient()
        if local_client.is_available():
            logger.info("Using Local LLM (ctransformers)")
            return local_client

        # 2. Try HuggingFace (free)
        hf_client = HuggingFaceClient()
        if hf_client.is_available():
            logger.info("Using HuggingFace Inference API")
            return hf_client
        
        # 3. Try OpenAI
        openai_client = OpenAIClient()
        if openai_client.is_available():
            logger.info("Using OpenAI API")
            return openai_client
        
        # Fallback
        logger.warning("No LLM provider configured, using fallback")
        return FallbackLLMClient()
    
    def create_medical_prompt(
        self,
        user_question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Create a medical chatbot prompt with context.
        
        Args:
            user_question: User's question
            context: Retrieved context from RAG
            conversation_history: Previous conversation messages
        
        Returns:
            Formatted prompt
        """
        # TinyLlama Chat Template
        # <|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>
        
        system_message = """You are a helpful medical assistant. Provide accurate, evidence-based medical information.
IMPORTANT:
- Answer in 2-3 SHORT sentences.
- Be precise and direct.
- Do NOT hallucinate conversation history.
- If unsure, recommend a doctor.
MEDICAL CONTEXT:
{context}"""

        # Add conversation history if available
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    history_text += f"<|user|>\n{content}</s>\n"
                else:
                    history_text += f"<|assistant|>\n{content}</s>\n"
        
        # Construct full prompt using template
        full_prompt = f"<|system|>\n{system_message.format(context=context)}</s>\n"
        
        if history_text:
            full_prompt += history_text
            
        full_prompt += f"<|user|>\n{user_question}</s>\n<|assistant|>\n"
        
        return full_prompt
    
    def generate_response(
        self,
        user_question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a medical response.
        
        Args:
            user_question: User's question
            context: Retrieved context from RAG
            conversation_history: Previous messages
            temperature: LLM temperature
        
        Returns:
            Generated response
        """
        prompt = self.create_medical_prompt(user_question, context, conversation_history)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=512
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return (
                "I apologize, but I encountered an error while generating a response. "
                "Please try again or rephrase your question."
            )
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.client.is_available()


# Singleton instance
_llm_client_instance = None


def get_llm_client() -> MedicalLLMClient:
    """Get or create LLM client singleton."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = MedicalLLMClient()
    return _llm_client_instance
