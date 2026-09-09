from app.database import get_plan_names


def find_matching_plans(question: str):
    plans = get_plan_names()
    normalized_question = question.lower()

    # First check whether the user explicitly mentioned
    # one or more complete plan names.
    full_matches = []

    for plan in plans:
        normalized_plan = plan.lower()

        if normalized_plan in normalized_question:
            full_matches.append(plan)

    # If full plan names are explicitly present,
    # the user's intention is considered clear.
    if full_matches:
        return {
            "matches": full_matches,
            "explicit": True,
        }

    # Otherwise check for partial references
    # such as "premium", "monthly", or "annual".
    partial_matches = []

    question_words = set(
        normalized_question.split()
    )

    for plan in plans:
        plan_words = set(
            plan.lower().split()
        )

        common_words = (
            question_words & plan_words
        )

        if common_words:
            partial_matches.append(plan)

    return {
        "matches": partial_matches,
        "explicit": False,
    }


def check_plan_ambiguity(question: str):
    result = find_matching_plans(
        question
    )

    matches = result["matches"]
    explicit = result["explicit"]

    # Explicit full plan names are not ambiguous,
    # even when more than one plan is mentioned.
    if explicit:
        return {
            "ambiguous": False,
            "matches": matches,
        }

    # A partial reference matching multiple plans
    # requires clarification.
    if len(matches) > 1:
        return {
            "ambiguous": True,
            "matches": matches,
        }

    return {
        "ambiguous": False,
        "matches": matches,
    }