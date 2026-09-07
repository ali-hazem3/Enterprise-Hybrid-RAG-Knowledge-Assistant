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
- Questions whose answer comes from the Noor Market handbook.
- This includes:
  managers
  branch managers
  opening hours
  return policy
  delivery rules
  loyalty program
  suppliers
  quality
  contact information
- IMPORTANT:
  If a question asks for a count, comparison, or number based on handbook
  information, it is still RAG.
- Examples:
  "Who manages Marina?" -> rag
  "How many branches have managers?" -> rag
  "How many branches are mentioned in the handbook?" -> rag

sql
- Questions whose answer requires structured business data from SQL.
- This includes:
  revenue
  sales
  quantities sold
  product performance
  best-selling products
  branch revenue
  numerical aggregations over sales data
- Examples:
  "Which branch generated the most revenue?" -> sql
  "How many units were sold?" -> sql
  "Which product sold the most?" -> sql

unknown
- Questions that cannot be answered using either the Noor Market handbook
  or Noor Market SQL database.

Choose the intent based on WHERE THE REQUIRED INFORMATION IS STORED,
not simply because the question asks for a number.

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