"""
Entry point for the Transformer-Based LLM Framework for Automated Feature Extraction.
"""
from experiments.run_experiments import run_all
from src.utils import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("Initializing Framework Sequence...")
    try:
        run_all()
    except Exception as e:
        logger.error(f"Framework execution failed: {e}", exc_info=True)
