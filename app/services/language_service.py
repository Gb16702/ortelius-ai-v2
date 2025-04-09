import logfire
from langdetect import detect, LangDetectException
from typing import Tuple

class LanguageService:
  @staticmethod
  def detect_language(text: str) -> Tuple[str, float]:
    try:
      print(text)
      if not text or not text.strip():
        logfire.warning("Text is empty")
        return "en", 1.0

      language = detect(text)

      logfire.info(f"Language detected: {language}")
      return language, 1.0

    except LangDetectException as e:
      logfire.error(f"Error detecting language: {str(e)}")
      return "en", .5
