from langchain_core.messages import trim_messages
from src.config import MAX_TOKENS_TRIMMER, TRIMMER_STRATEGY, TRIMMER_INCLUDE_SYSTEM, TRIMMER_ALLOW_PARTIAL, TRIMMER_START_ON
from src.config import LLM

# Initialize the message trimmer for managing chat history length
trimmer = trim_messages(
    max_tokens=MAX_TOKENS_TRIMMER,
    strategy=TRIMMER_STRATEGY,
    token_counter=LLM,
    include_system=TRIMMER_INCLUDE_SYSTEM,
    allow_partial=TRIMMER_ALLOW_PARTIAL,
    start_on=TRIMMER_START_ON,
)
