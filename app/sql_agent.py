from langchain_openai import AzureChatOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)

from app.database import (
    execute_select_query,
    get_database_schema,
)


llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


BUSINESS_RULES = """
Business rules:
- Revenue = dbo.sales.quantity * dbo.products.unit_price
"""


SQL_PROMPT = """
You are a SQL generation assistant for Microsoft SQL Server.

Generate one valid read-only T-SQL query that answers the CURRENT user's question.

Current database schema:

{schema}

{business_rules}

Conversation history:
{history}

Current user question:
{question}

Rules:
- Use the conversation history when the current question is a follow-up.
- Use only tables and columns from the provided schema.
- Use SELECT queries only.
- WITH common table expressions are allowed.
- Use JOINs when needed.
- Do not invent tables or columns.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE,
  CREATE, MERGE, EXEC, or EXECUTE.
- Return only the SQL query.
- Do not use markdown code fences.
- Do not explain the query.
"""


REPAIR_PROMPT = """
The previous SQL query failed.

Generate a corrected Microsoft SQL Server T-SQL query.

Current database schema:

{schema}

{business_rules}

Conversation history:
{history}

Original user question:
{question}

Failed SQL:
{query}

SQL Server error:
{error}

Rules:
- Correct the query using the SQL Server error.
- Use the conversation history when the question is a follow-up.
- Use only tables and columns from the provided schema.
- Generate read-only SQL only.
- Use SELECT or WITH.
- Use JOINs when needed.
- Do not invent tables or columns.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE,
  CREATE, MERGE, EXEC, or EXECUTE.
- Return only the corrected SQL query.
- Do not use markdown code fences.
- Do not explain anything.
"""


FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "MERGE",
    "EXEC",
    "EXECUTE",
}


def format_schema(schema: dict) -> str:
    lines = []

    for table_name, columns in schema.items():
        lines.append(
            f"Table: {table_name}"
        )

        for column in columns:
            lines.append(
                f"- {column['column_name']} "
                f"({column['data_type']})"
            )

        lines.append("")

    return "\n".join(lines)


def generate_sql(
    question: str,
    history: str = "",
) -> str:
    schema = get_database_schema()

    formatted_schema = format_schema(
        schema
    )

    prompt = SQL_PROMPT.format(
        schema=formatted_schema,
        business_rules=BUSINESS_RULES,
        history=history,
        question=question,
    )

    response = llm.invoke(prompt)

    return response.content.strip()


def validate_sql(query: str) -> bool:
    normalized_query = (
        query
        .strip()
        .upper()
    )

    if not (
        normalized_query.startswith("SELECT")
        or normalized_query.startswith("WITH")
    ):
        return False

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if keyword in normalized_query:
            return False

    return True


def repair_sql(
    question: str,
    query: str,
    error: str,
    history: str = "",
) -> str:
    schema = get_database_schema()

    formatted_schema = format_schema(
        schema
    )

    prompt = REPAIR_PROMPT.format(
        schema=formatted_schema,
        business_rules=BUSINESS_RULES,
        history=history,
        question=question,
        query=query,
        error=error,
    )

    response = llm.invoke(prompt)

    return response.content.strip()


def run_sql_agent(
    question: str,
    history: str = "",
    max_retries: int = 2,
):
    query = generate_sql(
        question=question,
        history=history,
    )

    for attempt in range(
        max_retries + 1
    ):
        if not validate_sql(query):
            raise ValueError(
                "Generated SQL was rejected "
                "by the safety validator."
            )

        try:
            results = execute_select_query(
                query
            )

            return {
                "query": query,
                "results": results,
                "attempts": attempt + 1,
            }

        except Exception as error:
            if attempt == max_retries:
                raise

            query = repair_sql(
                question=question,
                query=query,
                error=str(error),
                history=history,
            )