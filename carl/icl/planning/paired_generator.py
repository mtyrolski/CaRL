"""A matched two-slot intent generator with typed grounding and overlap checks."""

import json
from collections.abc import Sequence
from typing import Literal, TypeAlias, cast

from carl.icl.intent_domain import Intent, SokobanObservation, decode_intent, validate_intent_proposal
from carl.icl.intents.generation.data import parse_json
from carl.icl.planning.craftax_tasks import CraftaxIntent, intent_is_admissible
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import BoardRows, JSONObject, JSONValue


Arm: TypeAlias = Literal["unlabelled", "relative"]
Observation: TypeAlias = BoardRows | JSONObject
SCALES = (0.25, 0.75)
COMMON = (
    'Choose two distinct useful intents for the current task. Reply only as {"choices":["ID1","ID2"]}, '
    "using two different IDs from options. Each intent describes effects that are false now and conditions to preserve. "
    "Use the observation, public task and examples. Keep the reply short."
)
COPY_IDS = (
    "Choose two distinct useful intents from options for the current task. Return only a JSON array of two strings. "
    "Copy each string exactly from an id field in options. Use the observation, public task and examples. "
    "Choose effects that are false now with conditions that can be preserved."
)
VOCABULARY = {
    "sokoban": "Coordinates are zero-based row and column. # wall, - floor, . goal, $ box, * box on goal, @ player, "
               "+ player on goal. Intent predicates are box_at, player_at and boxes_on_goals_at_least.",
    "craftax": "Use visible inventory, facing and surroundings. Intent predicates are inventory_at_least and tool_tier_at_least.",
}
RELATIVE = (
    "Requested u values apply to the two choices in order. A larger u represents a longer successful intent "
    "for this controller. Use examples with measured u to interpret these requests."
)


def system_prompt(domain: str, arm: Arm, *, edition: int = 1) -> str:
    common = COMMON if edition == 1 else COPY_IDS
    return " ".join((common, VOCABULARY[domain], RELATIVE)) if arm == "relative" else " ".join((common, VOCABULARY[domain]))


def _intent_content(intent: JSONObject | Intent) -> JSONObject:
    return {"effects": intent["effects"], "preserve": intent["preserve"]}


def effect_id(intent: JSONObject | Intent) -> str:
    """Ignore wording, predicate order and preservation variants of the same effect."""
    effects = cast(Sequence[JSONValue], intent["effects"])
    return contract_hash(sorted(map(contract_hash, effects)))


def _visible(domain: str, observation: Observation) -> Observation:
    if domain == "sokoban":
        if not isinstance(observation, list):
            raise ValueError("Sokoban needs board rows")
        return observation
    if not isinstance(observation, dict):
        raise ValueError("Craftax needs a visible object")
    return {key: observation[key] for key in ("neighbors", "inventory", "facing", "health", "sleeping", "resting")
            if key in observation}


def messages(
    domain: str, arm: Arm, observation: Observation, task: str,
    support: Sequence[JSONObject], options: Sequence[JSONObject], *, edition: int = 1,
) -> list[dict[str, str]]:
    """Both arms see identical support and option content. Only the relative arm sees u."""
    if len(options) < len(SCALES):
        raise ValueError("Two slots require at least two distinct admissible options")
    measured = {row["measured_u"] for row in support if row.get("measured_u") is not None}
    if arm == "relative" and len(measured) < 2:
        raise ValueError("Relative generation requires at least two distinct measured scale labels")
    examples = []
    for row in support:
        example = {"observation": _visible(domain, row["observation"]), "task": row["task"],
                   "intent": _intent_content(row["intent"])}
        if arm == "relative" and row.get("measured_u") is not None:
            example["u"] = row["measured_u"]
        examples.append(example)
    query: JSONObject = {
        "examples": examples, "observation": _visible(domain, observation), "task": task,
        "options": [{"id": row["id"], **_intent_content(row["intent"])} for row in options],
    }
    if arm == "relative":
        query["requested_u"] = list(SCALES)
    return [{"role": "system", "content": system_prompt(domain, arm, edition=edition)},
            {"role": "user", "content": json.dumps(query, separators=(",", ":"), ensure_ascii=False, allow_nan=False)}]


def admissible(domain: str, intent: JSONValue | Intent, observation: Observation) -> JSONObject:
    """Check typed effects against the live model-visible start."""
    if isinstance(intent, dict):
        sketch: JSONObject = {**intent, "instruction": None}
        intent = sketch
    if domain == "sokoban":
        assert isinstance(observation, list)
        decoded = decode_intent(intent)
        errors = validate_intent_proposal(decoded, SokobanObservation(tuple(observation)))
        if errors:
            raise ValueError(", ".join(errors))
        return dict(decoded)
    assert isinstance(observation, dict)
    native = CraftaxIntent.model_validate(intent)
    if not intent_is_admissible(native, observation["inventory"]):
        raise ValueError("Craftax intent is already satisfied or has invalid preservation")
    return native.model_dump(mode="json")


def _choices(text: str) -> Sequence[JSONValue]:
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("```"):
        value = value[len("```json\n"):-3].strip()
    parsed: JSONValue = parse_json(value)
    if isinstance(parsed, dict):
        if set(parsed) in ({"choices"}, {"intents"}):
            parsed = next(iter(parsed.values()))
        elif "effects" in parsed:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list of two choices")
    return parsed


def proposals(
    domain: str, arm: Arm, text: str, observation: Observation, options: Sequence[JSONObject],
) -> JSONObject:
    """Validate each slot and count repeated semantic effects once."""
    ids = {row["id"]: row["intent"] for row in options}
    parse_error = None
    try:
        choices = _choices(text)
    except (TypeError, ValueError) as error:
        choices = []
        parse_error = str(error)
    seen = set()
    results = []
    for index, scale in enumerate(SCALES):
        row: JSONObject = {"index": index, "requested_u": scale if arm == "relative" else None,
                           "status": "missing" if parse_error is None else "invalid", "intent": None}
        if index < len(choices):
            choice = choices[index]
            try:
                h = ids[choice.strip().upper()] if isinstance(choice, str) else choice
                intent = admissible(domain, h, observation)
            except (KeyError, TypeError, ValueError) as error:
                row.update(status="invalid", error=str(error))
            else:
                key = effect_id(intent)
                row.update(intent=intent, effect_id=key, status="duplicate" if key in seen else "valid")
                seen.add(key)
        results.append(row)
    return {"slots": results, "extra_choices": max(0, len(choices) - len(SCALES)), "parse_error": parse_error,
            "distinct_effects": len(seen), "within_overlap": any(row["status"] == "duplicate" for row in results)}


def score(first_hit: int | None, endpoint_value: float | None, discount: float) -> float | None:
    """One scheduler score for successful assigned intents and observed failures."""
    if endpoint_value is None:
        return None
    return discount ** first_hit * endpoint_value if first_hit is not None else 0.0
