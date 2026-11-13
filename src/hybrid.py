"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).


FORMALIZER_PROMPT = """Translate the puzzle into IR JSON. Do not solve it. Return one JSON object and nothing else.

IR format:
{ir_spec}

Variables:
- Knights and knaves: one integer variable per person, domain [0, 1] (1 = knight, 0 = knave).
- Ordering puzzles: one integer variable per item, representing its position in [1, N].
- Multi-category puzzles: first identify all categories and their possible values from the puzzle. Create one position variable for every value in every category, all with domain [1, N].
- For multi-category puzzles, include every value from every category. Do not mix values from different categories or create helper variables representing combinations of values.
- Use descriptive names matching the entities and values. Never use arbitrary variable names or declare a variable twice.

all_different:
- Ordering puzzles: one group containing all items.
- Multi-category puzzles: one separate group per category, containing all its values. Never combine different categories into one group.
- Knights and knaves: do not use all_different. Set it to an empty list [].
- all_different is a separate IR field, never a constraint. Do not reproduce it through pairwise inequalities or enumerate possible assignments.

Constraints:
- One clue or quoted statement in, one constraint out. Preserve its exact meaning. Do not add, omit or duplicate conditions.
- If one clue contains multiple logically connected parts, represent them together in one constraint using and/or/not. Do not combine separate clues into one constraint.
- Use the simplest valid expression supported by the IR format. Prefer direct comparisons and logical operators (and, or, not).
- Do not wrap Boolean expressions in unnecessary arithmetic or comparisons with 1.
- Use sums of binary variables directly when counting knights. For example, exactly two knights means the sum of all five variables equals 2. Do not enumerate possible combinations.
- For a statement S made by person X, use exactly one constraint: X == (S). Do not add S as a separate constraint.
- When counting knights, include every person mentioned in the counted group, including the speaker when applicable.
- Do not enumerate permutations, assignments or truth tables. Express the original conditions directly through constraints on variables.
- For position relationships:
  * Fixed position: A == P
  * Next to: abs(A - B) == 1
  * Not next to: abs(A - B) > 1
  * Immediately to the left: A + 1 == B
  * Somewhere to the left: A < B
  * Exactly K items between: abs(A - B) == K + 1
  * Either end: A == 1 or A == N
  * Same entity across categories: A == B
  * Explicitly different attributes: A != B
- For clarity, exactly one item between A and B means abs(A - B) == 2. Exactly two items between A and B means abs(A - B) == 3.
- In multi-category puzzles, values belonging to the same entity must have equal positions. For example, if the person with the red door has a turtle, use Red == Turtle.
- Never put variable declarations, all_different, lists or unsupported function calls inside constraints. Each constraint must be one valid expression supported by the IR format.

Examples:
Example 1:
On an island, knights always tell the truth and knaves always lie.
You meet Taisiia, Ustym and Fedora.

Taisiia says: "All three of us are knaves."
Ustym says: "Exactly one of us three is a knight."
Fedora says: "Ustym is a knave."

IR:
{{
  "variables": {{
    "Taisiia": [0, 1],
    "Ustym": [0, 1],
    "Fedora": [0, 1]
  }},
  "all_different": [],
  "constraints": [
    "Taisiia == (Taisiia == 0 and Ustym == 0 and Fedora == 0)",
    "Ustym == (Taisiia + Ustym + Fedora == 1)",
    "Fedora == (Ustym == 0)"
  ]
}}

Example 2:
Five people stand in a queue at the canteen, positions 1 (front)
to 5 (back): Artem, Bohdana, Viktor, Halyna and Dmytro.

1. Viktor is somewhere in front of Halyna.
2. Artem stands directly behind Bohdana.
3. Dmytro is either at the very front or at the very back.
4. Halyna is not at the back.
5. Bohdana is not at the front.

What is the order of the queue?

IR:
{{
  "variables": {{
    "Artem": [1, 5],
    "Bohdana": [1, 5],
    "Viktor": [1, 5],
    "Halyna": [1, 5],
    "Dmytro": [1, 5]
  }},
  "all_different": [
    ["Artem", "Bohdana", "Viktor", "Halyna", "Dmytro"]
  ],
  "constraints": [
    "Viktor < Halyna",
    "Artem == Bohdana + 1",
    "Dmytro == 1 or Dmytro == 5",
    "Halyna != 5",
    "Bohdana != 1"
  ]
}}

Final check:
- Every category and every value from the answer format is represented.
- Every variable has the correct domain and belongs to the correct category.
- Every all_different group contains exactly the required variables.
- Every original clue is represented by the correct constraint.
- No original clue is duplicated or omitted.
- Do not remove or modify a clue because it appears redundant or contradictory. Preserve the original conditions even if they make the puzzle unsolvable.

If feedback is provided, correct the reported errors without changing previously correct parts of the IR.

Return only valid JSON. Do not include explanations or the final solution.

Puzzle:
{text}


{feedback}"""




def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)

def validate_ir(ir: dict) -> list[str]:
    errors: list[str] = []
    can_build = True

    if not isinstance(ir, dict):
        return ["IR must be a JSON object."]

    variables = ir.get("variables")
    if not isinstance(variables, dict) or not variables:
        return ['"variables" must be a non-empty object of variable names and domains.']

    domains: dict[str, list[tuple[int, int]]] = {}
    name_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for name, domain in variables.items():
        if not isinstance(name, str) or not name_pattern.fullmatch(name):
            errors.append(f"Invalid variable name {name!r}: use ASCII letters, digits and underscores; do not start with a digit.")
            can_build = False
            continue

        if isinstance(domain, list) and len(domain) == 2 and all(type(v) is int for v in domain) and domain[0] <= domain[1]:
            domains[name] = [(domain[0], domain[1])]
        elif isinstance(domain, dict) and set(domain) == {"values"} and isinstance(domain["values"], list) and domain["values"] and all(type(v) is int for v in domain["values"]):
            domains[name] = [(v, v) for v in set(domain["values"])]
        else:
            errors.append(
                f'Invalid domain for variable "{name}": {domain!r}. '
                'Use [low, high] with integers and low <= high, or {"values": [integer, ...]}.'
            )
            can_build = False

    groups = ir.get("all_different", [])
    if not isinstance(groups, list):
        errors.append('"all_different" must be a list of lists of variable names.')
        can_build = False
    else:
        for i, group in enumerate(groups):
            if not isinstance(group, list) or any(not isinstance(name, str) for name in group):
                errors.append(f"all_different[{i}] must be a list of variable names.")
                can_build = False
                continue
            if len(group) < 2:
                errors.append(f"all_different[{i}] = {group!r} has fewer than two variables; remove this group.")
                can_build = False
                continue
            if len(set(group)) != len(group):
                errors.append(f"all_different[{i}] contains duplicate variable names: {group!r}.")
                can_build = False
                continue
            unknown = [name for name in group if name not in variables]
            if unknown:
                errors.append(f"all_different[{i}] references undeclared variable(s): {unknown}.")
                can_build = False
                continue
            if all(name in domains for name in group):
                intervals = sorted(part for name in group for part in domains[name])
                merged: list[list[int]] = []
                for low, high in intervals:
                    if merged and low <= merged[-1][1] + 1:
                        merged[-1][1] = max(merged[-1][1], high)
                    else:
                        merged.append([low, high])
                available = sum(high - low + 1 for low, high in merged)
                if len(group) > available:
                    errors.append(
                        f"all_different[{i}] is impossible: {len(group)} variables have only "
                        f"{available} possible distinct values. Check that this is a one-of-each "
                        "category, not a group of independent yes/no variables."
                    )
                    can_build = False

    constraints = ir.get("constraints", [])
    if not isinstance(constraints, list) or any(not isinstance(c, str) for c in constraints):
        errors.append('"constraints" must be a list of expression strings.')
        can_build = False
    else:
        seen: dict[str, int] = {}
        for i, expression in enumerate(constraints):
            if not expression.strip():
                errors.append(f"constraints[{i}] is empty.")
                can_build = False
            elif expression in seen:
                errors.append(f"constraints[{i}] duplicates constraints[{seen[expression]}]: {expression!r}. Keep one copy.")
            else:
                seen[expression] = i

    objective = ir.get("objective")
    if objective is not None:
        if not isinstance(objective, dict) or len(objective) != 1 or next(iter(objective), None) not in {"maximize", "minimize"}:
            errors.append('"objective" must be exactly {"maximize": "<expression>"} or {"minimize": "<expression>"}.')
            can_build = False
        elif not isinstance(next(iter(objective.values())), str) or not next(iter(objective.values())).strip():
            errors.append('The objective expression must be a non-empty string.')
            can_build = False

    if can_build:
        try:
            ir_solver.build(ir)
        except (ValueError, SyntaxError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"ir_solver could not compile the expressions (unsupported syntax or unknown variable): {exc}")

    return errors


def build_retry_feedback(ir: dict, errors: list[str]) -> str:
    previous_ir = json.dumps(ir, ensure_ascii=False, separators=(",", ":"))
    error_details = "\n".join(f"- {error}" for error in errors)
    return (
        "The validator rejected your previous IR before it was sent to Z3.\n\n"
        "PREVIOUS IR (correct this version):\n"
        f"{previous_ir}\n\n"
        "VALIDATION ERRORS (fix each reported problem):\n"
        f"{error_details}\n\n"
        "Return the complete corrected JSON object. Preserve all parts that are not identified "
        "as erroneous. Correct unknown names instead of inventing helper variables. Use only "
        "the IR syntax from the specification (for example, abs() in lowercase; no lists, "
        "tuples or unsupported functions inside expressions). For an impossible all_different "
        "group, check whether its variables truly form a one-of-each category; do not apply it "
        "to knights/knaves or independent yes/no variables. Do not change, omit or add puzzle "
        "clues to force a unique or satisfiable result. This retry fixes validation errors only. "
        "Return JSON and nothing else."
    )




def run_solver(ir: dict) -> dict:

    if ir.get("objective") is not None:
        return ir_solver.solve(ir, limit=1)

    return ir_solver.solve(ir, limit=2)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present_knights_and_knaves(result: dict) -> dict:
    status = result["status"]

    if status == "none":
        return Answer(status="none").model_dump()

    assignment = result["solutions"][0]

    solution = {
        name: "knight" if value == 1 else "knave"
        for name, value in assignment.items()
    }

    return Answer(
        status=status,
        solution=solution
    ).model_dump()


def present(problem: dict, ir: dict, result: dict) -> dict:

    if "knights and knaves" in problem.get("title", "").lower():
        return present_knights_and_knaves(result)

    status = result["status"]

    if status == "none":
        return Answer(status="none").model_dump()

    prompt = PRESENT_PROMPT.format(
        text=problem["text"],
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(
            result["solutions"][0],
            ensure_ascii=False
        ),
        answer_format=problem["answer_format"]
    )

    parsed, raw = llm.ask_json(prompt, Answer)

    return Answer(
        status=status,
        solution=parsed.solution,
        objective=result.get("objective")
    ).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        feedback = build_retry_feedback(ir, errors)
        ir = formalize(problem, feedback=feedback)
        errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": errors, "feedback": feedback})

        if errors:
            raise HybridError(
                "invalid IR after one retry: " + "; ".join(errors)
            )

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
