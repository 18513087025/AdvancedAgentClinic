def build_system_prompt(
    profile: dict,
    runtime_info: dict | None = None,
) -> str:
    runtime_info = runtime_info or {}

    def _as_bullets(items) -> str:
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)

    parts: list[str] = []

    name = profile.get("name", "an AI agent")
    role = profile.get("role", "")
    goal = profile.get("goal", "")
    style = profile.get("style", "")
    constraints = profile.get("constraints", [])
    capabilities = profile.get("capabilities", [])
    input_format = profile.get("input_format")
    output_format = profile.get("output_format")
    termination_rule = profile.get("termination_rule")
    reminder = profile.get("reminder")

    parts.append(f"You are {name}.")
    if role:
        parts.append(f"Your role is: {role}")

    if goal:
        parts.extend([
            "",
            "[GOAL]",
            str(goal),
        ])

    if style:
        parts.extend([
            "",
            "[STYLE]",
            str(style),
        ])

    parts.extend([
        "",
        "[CONSTRAINTS]",
        _as_bullets(constraints),
    ])

    if capabilities:
        parts.extend([
            "",
            "[CAPABILITIES]",
            _as_bullets(capabilities),
        ])

    runtime_rules: list[str] = []
    if "max_turns" in runtime_info:
        runtime_rules.append(
            f"You may ask at most {runtime_info['max_turns']} questions total."
        )
    if "turns" in runtime_info:
        runtime_rules.append(
            f"You have asked {runtime_info['turns']} questions so far."
        )
    if "image_rule" in runtime_info:
        runtime_rules.append(str(runtime_info["image_rule"]))

    if runtime_rules:
        parts.extend([
            "",
            "[RUNTIME RULES]",
            _as_bullets(runtime_rules),
        ])

    if input_format:
        parts.extend([
            "",
            "[INPUT]",
            str(input_format),
        ])

    if output_format:
        parts.extend([
            "",
            "[OUTPUT]",
            str(output_format),
        ])

    if termination_rule:
        parts.extend([
            "",
            "[TERMINATION]",
            str(termination_rule),
        ])

    context = runtime_info.get("context")
    if context is not None:
        context_title = runtime_info.get("context_title", "CONTEXT")
        parts.extend([
            "",
            f"[{context_title}]",
            str(context),
        ])

    case_info = runtime_info.get("case_info")
    if case_info is not None:
        case_title = runtime_info.get("case_title", "CASE INFORMATION")
        parts.extend([
            "",
            f"[{case_title}]",
            str(case_info),
        ])

    if reminder:
        parts.extend([
            "",
            "[REMINDER]",
            str(reminder),
        ])
#     if runtime_info.get("is_last_turn"):
#         extra_rule = """
#         IMPORTANT:
#         This is the FINAL discussion turn.

#         You MUST do one of the following:
#         - If evidence is sufficient → output FINAL JSON
#         - If evidence is insufficient → STILL output your BEST possible structured JSON

#         DO NOT continue discussion.
#         DO NOT ask more questions.
#           """

    return "\n".join(parts).strip()