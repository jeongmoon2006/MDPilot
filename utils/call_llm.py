import os
import logging
from google import genai

logger = logging.getLogger(__name__)

def call_llm(prompt):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    response = client.models.generate_content(model=model, contents=prompt)
    logger.debug(f"LLM prompt length: {len(prompt)} chars")
    return response.text

if __name__ == "__main__":
    response = call_llm("In one sentence, what is RMSD in molecular dynamics?")
    print(response)
