from app.database import get_plan_names


def find_matching_plans(question: str):
    plans = get_plan_names()
    normalized_question = question.lower()

    # Step 1: Look for complete plan names first
    exact_matches = []

    for plan in plans:
        normalized_plan = plan.lower()

        if normalized_plan in normalized_question:
            exact_matches.append(plan)

    if exact_matches:
        return exact_matches

    # Step 2: Only use partial matching when
    # no complete plan name was found
    partial_matches = []

    question_words = set(
        normalized_question.split()
    )

    for plan in plans:
        plan_words = set(
            plan.lower().split()
        )

        common_words = (
            question_words
            & plan_words
        )

        if common_words:
            partial_matches.append(plan)

    return partial_matches


def check_plan_ambiguity(question: str):
    matches = find_matching_plans(question)

    if len(matches) <= 1:
        return {
            "ambiguous": False,
            "matches": matches,
        }

    return {
        "ambiguous": True,
        "matches": matches,
    }