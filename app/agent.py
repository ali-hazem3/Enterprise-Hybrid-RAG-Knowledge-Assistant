from langchain_openai import AzureChatOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)

from app.router import classify_intent
from app.rag import search_documents
from app.sql_agent import run_sql_agent
from app.ambiguity import check_plan_ambiguity

from app.memory import (
    get_conversation_history,
    save_interaction,
)

from app.logger import logger


llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


CONTEXTUALIZE_PROMPT = """
You rewrite follow-up questions into standalone questions.

Use the conversation history only to resolve references in the current question.

Examples of references include:
- it
- they
- them
- its
- that branch
- that plan
- that one
- the same one
- what about it

Do not answer the question.

Do not choose whether the question is SQL, RAG, or unknown.

Do not add information that is not present in the conversation.

If the current question is already standalone, return it unchanged.

Return ONLY the rewritten standalone question.

Conversation history:
{history}

Current question:
{question}
"""


RAG_RESPONSE_PROMPT = """
You are a PulseFit assistant.

Answer the user's question using ONLY the retrieved PulseFit
handbook context.

If the retrieved context does not contain enough information to answer,
say that the answer is not available in the PulseFit handbook.

Do not invent information.

Keep the answer clear and concise.

Include the source filename at the end of the answer.

User question:
{question}

Retrieved handbook context:
{context}

Source:
{source}
"""


SQL_RESPONSE_PROMPT = """
You are a PulseFit business data assistant.

Answer the user's question using ONLY the SQL query results.

Do not invent numbers or information.

Do not rename one metric as another metric.

Keep the answer clear and concise.

User question:
{question}

SQL results:
{results}
"""


def format_conversation_history(history) -> str:
    if not history:
        return "No previous conversation."

    lines = []

    for message in history:
        role = message["role"]
        content = message["content"]

        lines.append(
            f"{role.capitalize()}: {content}"
        )

    return "\n".join(lines)


def format_rag_context(documents) -> str:
    context_parts = []

    for document in documents:
        context_parts.append(
            document.page_content
        )

    return "\n\n".join(
        context_parts
    )


def contextualize_question(
    question: str,
    history: str,
) -> str:
    prompt = CONTEXTUALIZE_PROMPT.format(
        history=history,
        question=question,
    )

    response = llm.invoke(prompt)

    standalone_question = (
        response.content.strip()
    )

    logger.info(
        "Question contextualized | "
        f"original={question!r} | "
        f"standalone={standalone_question!r}"
    )

    return standalone_question


def answer_rag_question(
    question: str,
) -> str:
    logger.info(
        "RAG retrieval started | "
        f"question={question!r}"
    )

    documents = search_documents(
        question,
        k=3,
    )

    logger.info(
        "RAG retrieval completed | "
        f"question={question!r} | "
        f"chunks_retrieved={len(documents)}"
    )

    if not documents:
        logger.warning(
            "RAG retrieval returned no documents | "
            f"question={question!r}"
        )

        return (
            "I could not find enough information "
            "in the PulseFit handbook."
        )

    context = format_rag_context(
        documents
    )

    source = documents[0].metadata.get(
        "source",
        "PULSEFIT_HANDBOOK.txt",
    )

    prompt = RAG_RESPONSE_PROMPT.format(
        question=question,
        context=context,
        source=source,
    )

    response = llm.invoke(prompt)

    answer = response.content.strip()

    logger.info(
        "RAG response generated | "
        f"question={question!r} | "
        f"source={source!r} | "
        f"chunks_used={len(documents)}"
    )

    return answer


def answer_sql_question(
    question: str,
    history: str = "",
) -> str:
    logger.info(
        "SQL path started | "
        f"question={question!r}"
    )

    sql_output = run_sql_agent(
        question=question,
        history=history,
    )

    # 1. Unsupported metric
    if sql_output.get(
        "unsupported_metric"
    ):
        logger.warning(
            "Unsupported business metric | "
            f"question={question!r}"
        )

        return (
            "That metric cannot be calculated from the "
            "available PulseFit database because the "
            "required data is not stored."
        )

    # 2. SQL rejected by safety validator
    if sql_output.get(
        "validator_failed"
    ):
        logger.warning(
            "SQL rejected by safety validator | "
            f"question={question!r} | "
            f"query={sql_output.get('query')!r}"
        )

        return (
            "I could not safely execute the generated "
            "database query because it was rejected "
            "by the SQL safety validator."
        )

    # 3. SQL execution failed after repair limit
    if sql_output.get(
        "execution_failed"
    ):
        logger.error(
            "SQL execution failed after repair limit | "
            f"question={question!r} | "
            f"attempts={sql_output.get('attempts')}"
        )

        return (
            "I could not execute the database query "
            "successfully after the allowed repair "
            "attempts."
        )

    results = sql_output["results"]

    # 4. Valid query, but zero matching rows
    if not results:
        logger.info(
            "SQL query returned no rows | "
            f"question={question!r}"
        )

        return (
            "The database query completed successfully, "
            "but no matching data was found."
        )

    # 5. Successful SQL result
    prompt = SQL_RESPONSE_PROMPT.format(
        question=question,
        results=results,
    )

    response = llm.invoke(prompt)

    answer = response.content.strip()

    logger.info(
        "SQL response generated | "
        f"question={question!r} | "
        f"attempts={sql_output['attempts']}"
    )

    return answer


def run_agent(
    question: str,
    session_id: str,
) -> str:
    logger.info(
        "Agent request started | "
        f"session={session_id!r} | "
        f"question={question!r}"
    )

    try:
        history = get_conversation_history(
            session_id=session_id,
        )

        logger.info(
            "Conversation history loaded | "
            f"session={session_id!r} | "
            f"messages={len(history)}"
        )

        formatted_history = (
            format_conversation_history(
                history
            )
        )

        standalone_question = (
            contextualize_question(
                question=question,
                history=formatted_history,
            )
        )

        intent = classify_intent(
            question=standalone_question,
        )

        logger.info(
            "Intent classified | "
            f"session={session_id!r} | "
            f"intent={intent!r} | "
            f"standalone_question="
            f"{standalone_question!r}"
        )

        if intent == "rag":
            answer = answer_rag_question(
                question=standalone_question,
            )

        elif intent == "sql":
            ambiguity_result = (
                check_plan_ambiguity(
                    standalone_question
                )
            )

            if ambiguity_result["ambiguous"]:
                matches = (
                    ambiguity_result["matches"]
                )

                options = ", ".join(
                    matches
                )

                answer = (
                    "Your question could refer to "
                    "more than one membership plan: "
                    f"{options}. "
                    "Please specify which plan you mean."
                )

                logger.info(
                    "Ambiguous plan reference detected | "
                    f"session={session_id!r} | "
                    f"question={standalone_question!r} | "
                    f"matches={matches!r}"
                )

            else:
                answer = answer_sql_question(
                    question=standalone_question,
                    history=formatted_history,
                )

        else:
            answer = (
                "I can only answer questions "
                "using the PulseFit handbook "
                "or PulseFit business data."
            )

            logger.info(
                "Unknown intent handled | "
                f"session={session_id!r} | "
                f"question={standalone_question!r}"
            )

        save_interaction(
            session_id=session_id,
            original_user_question=question,
            contextualized_question=standalone_question,
            selected_route=intent,
            assistant_response=answer,
        )

        logger.info(
            "Conversation saved to memory | "
            f"session={session_id!r}"
        )

        logger.info(
            "Agent request completed | "
            f"session={session_id!r} | "
            f"intent={intent!r}"
        )

        return answer

    except Exception as error:
        logger.exception(
            "Agent request failed | "
            f"session={session_id!r} | "
            f"question={question!r} | "
            f"error={str(error)!r}"
        )

        raise