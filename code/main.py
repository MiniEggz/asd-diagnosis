"""Main module to run classification pipeline."""
from pipeline import AbideClassificationPipeline

if __name__ == "__main__":
    acp = AbideClassificationPipeline()
    acp.run()

