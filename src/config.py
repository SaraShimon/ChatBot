from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# LLM and Embeddings models
LLM = init_chat_model("gpt-4o-mini", model_provider="openai")
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")


project_dir = './'


# Document processing settings
PDF_DATA_PATH = project_dir + "Data/Rag/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Trimmer settings (for chat history management)
MAX_TOKENS_TRIMMER = 2000
TRIMMER_STRATEGY = "last"
TRIMMER_INCLUDE_SYSTEM = True
TRIMMER_ALLOW_PARTIAL = False
TRIMMER_START_ON = "human"

# Path data files for tools
USER_VENDORS_FILE = Path(project_dir + "Data/DB/users_vendors.json")
USERS_DATA_FILE = Path(project_dir + "Data/DB/users_data.json")


# Path for human service queue
QUEUE_FILE = project_dir + "Data/DB/global_service_queue.json"
