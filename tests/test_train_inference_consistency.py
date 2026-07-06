"""
The GRU model is trained on features produced by extract_features.py, but at
inference time realtime_detector.py re-implements the same optical-flow
feature extraction from scratch. If the two implementations diverge, the
model sees a different feature distribution live than it was trained on --
independent of whether either implementation is individually 'correct'. This
file tests that the two pipelines behave the same way on identical synthetic
input, which is what actually matters for the model's real-world accuracy
claims in Chapter 3.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow", reason="realtime_detector.py imports tensorflow at module scope")

import extract_features as ef
from realtime_detector import OpticalFlowExtractor


def _write_video(tmp_path, frames, fps=30):
    path = tmp_path / "synthetic.mp4"
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


class TestKeypointRefreshParity:
    def test_refresh_uses_same_baseline_frame_in_both_pipelines(self, translating_square_sequence):
        """extract_features.py refreshes lost keypoints against `old_gray` (the
        previous frame); realtime_detector.py currently refreshes against
        `frame_gray` (the current frame). These must match, or the live
        detector is feeding the GRU model data it was never trained on.
        This test documents the expected (training-side) behavior and will
        fail against the current realtime_detector.py implementation."""
        frames = translating_square_sequence

        # --- training-side feature extraction logic (ground truth) ---
        old_gray = frames[0]
        old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **ef.feature_params)
        train_features = []
        for frame_gray in frames[1:]:
            if old_points is None or len(old_points) < 10:
                # training code re-detects against the PREVIOUS frame (old_gray)
                old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **ef.feature_params)
                if old_points is None:
                    train_features.append(np.zeros(8, dtype=np.float32))
                    old_gray = frame_gray.copy()
                    continue
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **ef.lk_params)
            train_features.append(ef.compute_motion_features(old_points, new_points, status))
            if new_points is not None and status is not None:
                good_new = new_points[status == 1]
                old_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else cv2.goodFeaturesToTrack(frame_gray, mask=None, **ef.feature_params)
            old_gray = frame_gray.copy()

        # --- realtime inference-side logic ---
        extractor = OpticalFlowExtractor()
        extractor.process_frame(frames[0])
        realtime_features = []
        for frame_gray in frames[1:]:
            features, _, _ = extractor.process_frame(frame_gray)
            realtime_features.append(features)

        train_features = np.array(train_features)
        realtime_features = np.array(realtime_features)

        # Not expecting bit-identical floats (different code paths), but the
        # tracked point COUNT (feature index 4) should follow the same trend --
        # both should stay high on a continuously-textured moving scene.
        train_counts = train_features[:, 4]
        realtime_counts = realtime_features[:, 4]

        assert np.mean(realtime_counts) > 0, "Realtime pipeline lost all tracked points"
        assert np.mean(realtime_counts) == pytest.approx(np.mean(train_counts), rel=0.5), (
            f"Training pipeline tracks ~{np.mean(train_counts):.1f} pts/frame on average, "
            f"realtime pipeline tracks ~{np.mean(realtime_counts):.1f} -- the live feature "
            f"distribution the GRU sees diverges significantly from training data."
        )
