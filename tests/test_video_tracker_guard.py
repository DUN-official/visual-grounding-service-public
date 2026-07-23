from grounding.video.tracker import OpenCVBoxTracker


def test_tracker_geometry_accepts_small_motion():
    tracker = OpenCVBoxTracker()
    assert tracker._geometry_is_valid(
        (100.0, 100.0, 150.0, 150.0),
        (105.0, 102.0, 156.0, 153.0),
    )


def test_tracker_geometry_rejects_large_scale_jump():
    tracker = OpenCVBoxTracker()
    assert not tracker._geometry_is_valid(
        (100.0, 100.0, 150.0, 150.0),
        (80.0, 80.0, 300.0, 300.0),
    )


def test_tracker_geometry_rejects_far_jump():
    tracker = OpenCVBoxTracker()
    assert not tracker._geometry_is_valid(
        (100.0, 100.0, 150.0, 150.0),
        (500.0, 500.0, 550.0, 550.0),
    )
