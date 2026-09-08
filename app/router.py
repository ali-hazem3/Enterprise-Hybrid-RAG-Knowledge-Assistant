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
You are an intent classifier for PulseFit.

Classify the user's question into exactly one of these three intents:

rag
- Questions whose answer comes from the PulseFit handbook.
- This includes:
  club managers
  opening hours
  ladies-only hours
  membership freeze policies
  refund policies
  guest policies
  guest passes
  personal training policies
  contact information
  general handbook information
- IMPORTANT:
  If a question asks for a count, comparison, or number based on handbook
  information, it is still RAG.
- Examples:
  "Who manages the Downtown club?" -> rag
  "What are the opening hours?" -> rag
  "Can Day Pass holders freeze their membership?" -> rag
  "How many free guest passes do annual members get?" -> rag
  "What is the refund policy for annual plans?" -> rag

sql
- Questions whose answer requires structured data from the PulseFit
  SQL database.
- This includes:
  membership plans
  plan prices
  billing types
  members
  member counts
  branches associated with members
  join dates
  payments
  payment dates
  revenue
  numerical aggregations over member or payment data
- Examples:
  "How many Premium Annual members are there?" -> sql
  "How many members belong to each branch?" -> sql
  "What is the price of Premium Monthly?" -> sql
  "How much revenue has PulseFit received?" -> sql
  "How many payments were made?" -> sql

unknown
- Questions that cannot be answered using either the PulseFit handbook
  or the PulseFit SQL database.

Choose the intent based on WHERE THE REQUIRED INFORMATION IS STORED,
not simply because the question asks for a number.

Do not attempt to answer the question.

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