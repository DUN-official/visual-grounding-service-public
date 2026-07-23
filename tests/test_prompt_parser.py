from grounding.task_parser import parse_grounding_prompt


def test_parse_find_bag_on_table():
    parsed = parse_grounding_prompt("find the bag on the table")
    assert parsed.target_object == "bag"
    assert parsed.location_hint == "on the table"
    assert parsed.action == "find"
    assert parsed.relation_detected is True


def test_parse_simple_open_vocabulary_target():
    parsed = parse_grounding_prompt("find the elevator control panel")
    assert parsed.target_object == "elevator control panel"
    assert parsed.location_hint is None
    assert parsed.action == "find"
