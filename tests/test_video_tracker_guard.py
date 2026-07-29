from grounding.video.tracker import OpenCVBoxTracker


class _DriftingTracker:
    def update(self, frame):
        return True, (75, 40, 30, 30)


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


def test_appearance_guard_recovers_original_target_instead_of_drifting():
    import numpy as np

    frame = np.zeros((140, 160, 3), dtype=np.uint8)
    pattern = np.indices((30, 30)).sum(axis=0) % 2
    frame[40:70, 40:70] = (pattern[:, :, None] * 255).astype(np.uint8)

    tracker = OpenCVBoxTracker(
        context_padding=0.0,
        minimum_appearance_similarity=0.55,
    )
    tracker._tracker = _DriftingTracker()
    tracker._previous_tracking_box = (40.0, 40.0, 70.0, 70.0)
    tracker._template = tracker._crop_gray(
        frame,
        tracker._previous_tracking_box,
    )

    success, box = tracker.update(frame)

    assert success is True
    assert tracker.last_source == "template_recovery"
    assert box == (40.0, 40.0, 70.0, 70.0)
    assert tracker.last_appearance_similarity >= 0.55
