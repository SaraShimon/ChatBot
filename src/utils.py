from langchain_core.messages import trim_messages
from src.config import MAX_TOKENS_TRIMMER, TRIMMER_STRATEGY, TRIMMER_INCLUDE_SYSTEM, TRIMMER_ALLOW_PARTIAL, TRIMMER_START_ON
from src.config import LLM
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator

# Initialize the message trimmer for managing chat history length
trimmer = trim_messages(
    max_tokens=MAX_TOKENS_TRIMMER,
    strategy=TRIMMER_STRATEGY,
    token_counter=LLM,
    include_system=TRIMMER_INCLUDE_SYSTEM,
    allow_partial=TRIMMER_ALLOW_PARTIAL,
    start_on=TRIMMER_START_ON,
)


def translate_heb_to_eng(text: str) -> str:
    try:
        lang_code = detect(text)
    except LangDetectException as e:
        print(f"Language detection failed for text: '{text}' - Error: {e}")
        return "Translation failed."
    if lang_code == 'en':
        return text
    if lang_code == 'he':
        try:
            # Use source='auto' to let GoogleTranslator detect the source language automatically
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text
        except Exception as e:
            print(f"Error during translation with deep_translator: {e}")
            return "Translation failed."
