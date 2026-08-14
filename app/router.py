from langchain_openai import AzureChatOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)


llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


ROUTER_PROMPT = """
You are an intent classifier for Noor Market.

Classify the user's question into exactly one of these three intents:

rag
- Questions about information found in the Noor Market handbook.
- Examples:
  managers
  opening hours
  return policy
  delivery rules
  loyalty program
  suppliers
  quality
  contact information

sql
- Questions that require structured business data from the SQL database.
- Examples:
  revenue
  sales
  quantities
  best-selling products
  branch performance
  numerical comparisons
  aggregations

unknown
- Questions that cannot be answered using either the Noor Market handbook
  or the Noor Market SQL database.

Return ONLY one word:

rag
sql
unknown

Do not explain your answer.

User question:
{question}
"""


def classify_intent(question: str) -> str:
    prompt = ROUTER_PROMPT.format(
        question=question,
    )

    response = llm.invoke(prompt)

    intent = response.content.strip().lower()

    allowed_intents = {
        "rag",
        "sql",
        "unknown",
    }

    if intent not in allowed_intents:
        return "unknown"

    return intent