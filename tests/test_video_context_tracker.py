from grounding.video.tracker import OpenCVBoxTracker


def test_context_box_is_larger_than_target():
    tracker = OpenCVBoxTracker(context_padding=0.5)
    target = (100.0, 100.0, 140.0, 130.0)
    context = tracker._context_box(target, 640, 480)
    assert context[0] < target[0]
    assert context[1] < target[1]
    assert context[2] > target[2]
    assert context[3] > target[3]


def test_target_round_trip_through_context_transform():
    tracker = OpenCVBoxTracker(context_padding=0.4)
    target = (100.0, 120.0, 180.0, 200.0)
    context = tracker._context_box(target, 640, 480)
    fraction = tracker._fraction_inside(target, context)
    restored = tracker._target_from_context(context, fraction, 640, 480)
    for expected, actual in zip(target, restored):
        assert abs(expected - actual) < 1e-6
