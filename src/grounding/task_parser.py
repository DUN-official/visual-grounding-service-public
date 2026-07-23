"""Deterministic structured parser for visual-grounding instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import QuantityIntent, ReasoningComplexity, SpatialConstraint


ACTION_PATTERNS = (
    ("track", "find"), ("follow", "find"), ("monitor", "find"),
    ("pick up", "pickup"), ("pickup", "pickup"), ("drop off", "dropoff"),
    ("deliver", "deliver"), ("approach", "approach"), ("look for", "find"),
    ("locate", "find"), ("identify", "find"), ("show me", "find"),
    ("where is", "find"), ("where are", "find"), ("find", "find"),
    ("ground", "find"),
)

RELATION_PATTERNS = (
    "to the left of", "to the right of", "on top of", "in front of",
    "at the end of", "next to", "closest to", "furthest from",
    "beside", "between", "opposite", "behind", "under", "below",
    "above", "near", "inside", "outside", "by", "on",
)

RELATION_CANONICAL = {
    "to the left of": "left_of", "to the right of": "right_of",
    "on top of": "on", "in front of": "in_front_of",
    "at the end of": "at_end_of", "next to": "beside",
    "closest to": "closest_to", "furthest from": "furthest_from",
    "beside": "beside", "between": "between", "opposite": "opposite",
    "behind": "behind", "under": "below", "below": "below",
    "above": "above", "near": "near", "inside": "inside",
    "outside": "outside", "by": "beside", "on": "on",
}

COLORS = {
    "black", "blue", "brown", "cyan", "gold", "gray", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "teal", "white",
    "yellow",
}
ATTRIBUTE_WORDS = COLORS | {
    "big", "large", "small", "tiny", "tall", "short", "long", "wide",
    "narrow", "open", "closed", "bright", "dark", "striped", "plain",
    "wooden", "metal", "plastic", "cardboard", "first", "second", "third",
    "last", "nearest", "closest", "furthest", "leftmost", "rightmost",
}
QUANTITY_WORDS = {
    "one": 1, "single": 1, "a": 1, "an": 1,
    "two": 2, "both": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
ALL_WORDS = {"all", "every", "each"}
IRREGULAR_SINGULAR = {
    "people": "person", "men": "man", "women": "woman", "children": "child",
    "boxes": "box", "packages": "package", "bags": "bag", "chairs": "chair",
    "doors": "door", "panels": "panel", "buttons": "button", "bottles": "bottle",
}
LEADING_DETERMINERS = ("the ", "a ", "an ")
TRAILING_POLITENESS = (" please", " for me")


@dataclass(frozen=True)
class ParsedGroundingPrompt:
    target_object: str | None
    target_phrase: str | None
    location_hint: str | None
    action: str | None
    quantity: QuantityIntent = QuantityIntent.UNKNOWN
    minimum_count: int = 1
    maximum_results: int = 1
    attributes: list[str] = field(default_factory=list)
    relations: list[SpatialConstraint] = field(default_factory=list)
    anchor_objects: list[str] = field(default_factory=list)
    relation_detected: bool = False
    reasoning_complexity: ReasoningComplexity = ReasoningComplexity.SIMPLE
    requires_guided_reasoning: bool = False
    guidance_reasons: list[str] = field(default_factory=list)


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def parse_grounding_prompt(instruction: str) -> ParsedGroundingPrompt:
    text = normalize_text(instruction)
    action, content = _remove_action(text)
    content = _remove_politeness(content)

    relations, first_relation_start = _extract_relations(content)
    target_segment = content if first_relation_start is None else content[:first_relation_start]

    target_segment, tail_attributes = _extract_attribute_tail(target_segment)
    quantity, minimum_count, target_segment = _extract_quantity(target_segment)
    target_phrase = _clean_phrase(target_segment)
    target_object, descriptor_attributes = _extract_target_and_attributes(target_phrase)
    attributes = _unique([*descriptor_attributes, *tail_attributes])

    if quantity == QuantityIntent.UNKNOWN and target_phrase:
        final_word = target_phrase.split()[-1]
        if _looks_plural(final_word):
            quantity = QuantityIntent.MULTIPLE
            minimum_count = max(2, minimum_count)
        else:
            quantity = QuantityIntent.ONE

    maximum_results = 10 if quantity in {QuantityIntent.MULTIPLE, QuantityIntent.ALL} else 1
    anchors = _unique([constraint.anchor for constraint in relations if constraint.anchor])
    location_hint = " ".join(c.raw_text for c in relations).strip() or None

    reasons = []
    if relations:
        reasons.append("spatial_relation")
    if attributes:
        reasons.append("visual_attribute")
    if quantity in {QuantityIntent.MULTIPLE, QuantityIntent.ALL}:
        reasons.append("multiple_objects")
    if any(a in ATTRIBUTE_WORDS for a in attributes):
        reasons.append("instance_disambiguation")
    if len(relations) > 1:
        reasons.append("multiple_relations")

    requires_guided = bool(reasons)
    constraint_count = len(relations) + len(attributes)
    if constraint_count > 1 or len(relations) > 1:
        complexity = ReasoningComplexity.MULTI_CONSTRAINT
    elif requires_guided:
        complexity = ReasoningComplexity.GUIDED
    else:
        complexity = ReasoningComplexity.SIMPLE

    return ParsedGroundingPrompt(
        target_object=target_object,
        target_phrase=target_phrase,
        location_hint=location_hint,
        action=action,
        quantity=quantity,
        minimum_count=minimum_count,
        maximum_results=maximum_results,
        attributes=attributes,
        relations=relations,
        anchor_objects=anchors,
        relation_detected=bool(relations),
        reasoning_complexity=complexity,
        requires_guided_reasoning=requires_guided,
        guidance_reasons=_unique(reasons),
    )


def has_relational_language(instruction: str, location_hint: str | None = None) -> bool:
    parsed = parse_grounding_prompt(f"{instruction} {location_hint or ''}")
    return parsed.relation_detected


def requires_guided_reasoning(instruction: str) -> bool:
    return parse_grounding_prompt(instruction).requires_guided_reasoning


def infer_target_from_vocabulary(instruction: str, vocabulary: list[str]) -> str | None:
    text = normalize_text(instruction)
    matches = [term for term in vocabulary if normalize_text(term) in text]
    return max(matches, key=len) if matches else None


def singularize_phrase(value: str) -> str:
    words = normalize_text(value).split()
    if not words:
        return ""
    words[-1] = _singularize_word(words[-1])
    return " ".join(words)


def _remove_action(text: str):
    for phrase, canonical in ACTION_PATTERNS:
        if text == phrase or text.startswith(phrase + " "):
            return canonical, text[len(phrase):].strip()
    return None, text


def _remove_politeness(text: str):
    for suffix in TRAILING_POLITENESS:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
    return text


def _extract_relations(text: str):
    alternatives = "|".join(re.escape(x) for x in sorted(RELATION_PATTERNS, key=len, reverse=True))
    matches = list(re.finditer(rf"\b({alternatives})\b", text))
    if not matches:
        return [], None
    constraints = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        anchor = _clean_phrase(text[match.end():end])
        raw = text[match.start():end].strip()
        if anchor:
            constraints.append(SpatialConstraint(
                relation=RELATION_CANONICAL[match.group(1)],
                anchor=singularize_phrase(anchor),
                raw_text=raw,
            ))
    return constraints, matches[0].start()


def _extract_attribute_tail(text: str):
    attributes = []
    # "man in green", "person wearing red", "bag with blue stripes"
    pattern = re.compile(r"\b(?:wearing|with|in)\s+([a-z][a-z\s-]*)$")
    match = pattern.search(text)
    if match:
        tail = normalize_text(match.group(1))
        tail_tokens = [token for token in re.findall(r"[a-z]+", tail) if token in ATTRIBUTE_WORDS]
        if tail_tokens:
            attributes.extend(tail_tokens)
            text = text[:match.start()].strip()
    return text, attributes


def _extract_quantity(text: str):
    words = text.split()
    if not words:
        return QuantityIntent.UNKNOWN, 1, text
    first = words[0]
    if first in ALL_WORDS:
        return QuantityIntent.ALL, 2, " ".join(words[1:])
    if first.isdigit():
        count = max(1, int(first))
        intent = QuantityIntent.ONE if count == 1 else QuantityIntent.MULTIPLE
        return intent, count, " ".join(words[1:])
    if first in QUANTITY_WORDS:
        count = QUANTITY_WORDS[first]
        intent = QuantityIntent.ONE if count == 1 else QuantityIntent.MULTIPLE
        return intent, count, " ".join(words[1:])
    return QuantityIntent.UNKNOWN, 1, text


def _extract_target_and_attributes(target_phrase: str | None):
    if not target_phrase:
        return None, []
    tokens = target_phrase.split()
    attributes = [token for token in tokens if token in ATTRIBUTE_WORDS]
    target_tokens = [token for token in tokens if token not in ATTRIBUTE_WORDS]
    if not target_tokens:
        target_tokens = tokens
    target = singularize_phrase(" ".join(target_tokens))
    return target or None, attributes


def _clean_phrase(text: str):
    value = normalize_text(text).strip(" .,:;!?")
    for determiner in LEADING_DETERMINERS:
        if value.startswith(determiner):
            value = value[len(determiner):].strip()
            break
    return value or None


def _looks_plural(word: str):
    word = normalize_text(word)
    if word in IRREGULAR_SINGULAR:
        return True
    return len(word) > 3 and word.endswith("s") and not word.endswith("ss")


def _singularize_word(word: str):
    word = normalize_text(word)
    if word in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))
