# utils.py
import logging
import time
from typing import TypeVar, Dict, Any, Optional
from functools import wraps

T = TypeVar('T')  # For generic type hints

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def time_it(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def extract_json(text: str) -> str:
    """Extract JSON object from text, handling potential non-JSON content.
    
    Args:
        text (str): Input text that may contain a JSON object
        
    Returns:
        str: Extracted JSON string or original text if no JSON found
    """
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text

def safe_get(data: Dict[str, Any], key: str, default: Optional[T] = None) -> Optional[T]:
    """Safely get a value from a dictionary with type hinting.
    
    Args:
        data (Dict[str, Any]): Dictionary to get value from
        key (str): Key to look up
        default (Optional[T]): Default value if key not found
        
    Returns:
        Optional[T]: Value from dictionary or default
    """
    return data.get(key, default)

# State type hints for workflow nodes
WorkflowState = Dict[str, Any]

logger = setup_logging()  # Initialize logger