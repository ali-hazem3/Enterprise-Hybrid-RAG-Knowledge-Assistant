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
from app.logger import logger

llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


BUSINESS_RULES = """
PulseFit business rules:

- Actual revenue = SUM(dbo.payments.amount).
- Plan price comes from dbo.plans.price.
- A member's branch is determined by joining:
  dbo.members.branch_id = dbo.branches.branch_id.
- A member's membership plan is determined by joining:
  dbo.members.plan_id = dbo.plans.plan_id.
- Member counts are calculated from dbo.members.
- Revenue by branch requires joining:
  dbo.payments -> dbo.members -> dbo.branches.
- Revenue by plan requires joining:
  dbo.payments -> dbo.members -> dbo.plans.
- Membership join-date questions use dbo.members.join_date.
- Payment-date questions use dbo.payments.pay_date.
- Plan billing type comes from dbo.plans.billing.

Important:
- Do not assume every member has made a payment.
- Do not calculate actual revenue using plan price multiplied by member count.
- Profit cannot be calculated because the database contains no cost or
  expense data.
- Active-member counts cannot be calculated because the database contains
  no membership status column.
- Do not invent unavailable metrics or substitute another metric.
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
- Before generating SQL, verify that all columns required to calculate the
  requested metric actually exist in the provided schema.
- If the requested metric cannot be calculated from the available schema,
  return exactly:
  UNSUPPORTED_METRIC
- Never substitute a different metric for the one requested.
- Return only the SQL query or UNSUPPORTED_METRIC.
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
- If the requested metric cannot actually be calculated from the schema,
  return exactly:
  UNSUPPORTED_METRIC
- Never substitute another metric for the requested metric.
- Return only the corrected SQL query or UNSUPPORTED_METRIC.
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

    if normalized_query == "UNSUPPORTED_METRIC":
        return True

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
    max_retries: int = 3,
):
    query = generate_sql(
        question=question,
        history=history,
    )

    logger.info(
        "SQL generated | "
        f"question={question!r} | "
        f"query={query!r}"
    )

    if query.strip().upper() == "UNSUPPORTED_METRIC":
        logger.warning(
            "Unsupported SQL metric | "
            f"question={question!r}"
        )

        return {
            "query": None,
            "results": [],
            "attempts": 0,
            "unsupported_metric": True,
            "validator_failed": False,
            "execution_failed": False,
        }

    for attempt in range(
        max_retries + 1
    ):
        attempt_number = attempt + 1

        is_valid = validate_sql(query)

        logger.info(
            "SQL validator result | "
            f"attempt={attempt_number} | "
            f"valid={is_valid} | "
            f"query={query!r}"
        )

        if not is_valid:
            logger.warning(
                "SQL rejected by validator | "
                f"attempt={attempt_number} | "
                f"query={query!r}"
            )

            return {
                "query": query,
                "results": [],
                "attempts": attempt_number,
                "unsupported_metric": False,
                "validator_failed": True,
                "execution_failed": False,
            }

        try:
            results = execute_select_query(
                query
            )

            logger.info(
                "SQL execution successful | "
                f"attempt={attempt_number} | "
                f"rows={len(results)}"
            )

            return {
                "query": query,
                "results": results,
                "attempts": attempt_number,
                "unsupported_metric": False,
                "validator_failed": False,
                "execution_failed": False,
            }

        except Exception as error:
            logger.error(
                "SQL execution error | "
                f"attempt={attempt_number} | "
                f"query={query!r} | "
                f"error={str(error)!r}"
            )

            if attempt == max_retries:
                logger.error(
                    "SQL repair limit reached | "
                    f"total_attempts={attempt_number}"
                )

                return {
                    "query": query,
                    "results": [],
                    "attempts": attempt_number,
                    "unsupported_metric": False,
                    "validator_failed": False,
                    "execution_failed": True,
                    "error": str(error),
                }

            repair_number = attempt + 1

            logger.info(
                "SQL repair started | "
                f"repair_attempt={repair_number}"
            )

            query = repair_sql(
                question=question,
                query=query,
                error=str(error),
                history=history,
            )

            logger.info(
                "SQL repaired | "
                f"repair_attempt={repair_number} | "
                f"query={query!r}"
            )

            if (
                query
                .strip()
                .upper()
                == "UNSUPPORTED_METRIC"
            ):
                logger.warning(
                    "Repair returned unsupported metric | "
                    f"repair_attempt={repair_number}"
                )

                return {
                    "query": None,
                    "results": [],
                    "attempts": attempt_number + 1,
                    "unsupported_metric": True,
                    "validator_failed": False,
                    "execution_failed": False,
                }