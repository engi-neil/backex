"""Shared optical-flow point tracking + motion feature extraction.

Used identically by both extract_features.py (offline training data
generation) and realtime_detector.py (live inference), so the two pipelines
can no longer silently diverge -- previously each maintained its own copy of
this logic, and the two copies drifted apart (see git history / the
train/inference consistency test).

The underlying algorithms are unchanged: Shi-Tomasi corner detection
(cv2.goodFeaturesToTrack) seeds points, Lucas-Kanade pyramidal optical flow
(cv2.calcOpticalFlowPyrLK) tracks them frame to frame. This module adds an
ROI layer (see roi.py) that constrains where those calls are allowed to
search and which resulting points are kept, to avoid tracking stray points
on background clutter.
"""
import cv2
import numpy as np

from roi import build_static_roi, combined_roi, points_inside_mask

FEATURE_DIM = 8

FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def _filter_by_status(old_points, new_points, status):
    """Applies the LK status mask, dropping OpenCV's redundant middle
    dimension so callers get plain (K, 2) arrays instead of (K, 1, 2)."""
    if old_points is None or new_points is None or status is None:
        return None, None

    status = status.reshape(-1)
    good_new = new_points[status == 1].reshape(-1, 2)
    good_old = old_points[status == 1].reshape(-1, 2)

    if len(good_new) == 0:
        return None, None

    return good_new, good_old


def _feature_stats(good_old, good_new):
    """Extracts the 8 motion features required by the GRU model."""
    if good_new is None or len(good_new) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    displacement = good_new - good_old
    dx = displacement[:, 0]
    dy = displacement[:, 1]

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    direction = np.arctan2(dy, dx)

    return np.array([
        np.mean(dx),
        np.mean(dy),
        np.mean(magnitude),
        np.mean(direction),
        len(good_new),
        np.std(dx),
        np.std(dy),
        np.std(magnitude)
    ], dtype=np.float32)


class OpticalFlowTracker:
    """Stateful Lucas-Kanade point tracker with ROI-constrained keypoint
    search. Both the training feature extractor and the live detector
    instantiate this class rather than reimplementing tracking themselves."""

    def __init__(self, feature_params=None, lk_params=None):
        self.feature_params = feature_params or FEATURE_PARAMS
        self.lk_params = lk_params or LK_PARAMS
        self.old_gray = None
        self.old_points = None
        self.static_roi = None

    def _search_mask(self, frame_shape, tracked_points=None):
        if self.static_roi is None:
            self.static_roi = build_static_roi(frame_shape)
        if tracked_points is None:
            tracked_points = self.old_points
        return combined_roi(frame_shape, tracked_points=tracked_points, static_mask=self.static_roi)

    def process_frame(self, frame_gray):
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        good_new = None
        good_old = None

        if self.old_points is None or len(self.old_points) < 10:
            mask = self._search_mask(frame_gray.shape)
            self.old_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=mask,
                **self.feature_params
            )
            # This frame is now the baseline these points were detected on --
            # don't run LK against a stale self.old_gray from a previous frame.
            self.old_gray = frame_gray.copy()
            return features, good_new, good_old

        if self.old_gray is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.old_gray,
                frame_gray,
                self.old_points,
                None,
                **self.lk_params
            )

            try:
                good_new, good_old = _filter_by_status(self.old_points, new_points, status)

                if good_new is not None and len(good_new) > 0:
                    # Drop any point that drifted outside the current ROI --
                    # tracking error or genuine exit from frame shouldn't
                    # leave a stray point contaminating future frames.
                    mask = self._search_mask(frame_gray.shape, tracked_points=good_new)
                    inside = points_inside_mask(good_new, mask)
                    good_new, good_old = good_new[inside], good_old[inside]
                    if len(good_new) == 0:
                        good_new, good_old = None, None

                features = _feature_stats(good_old, good_new)

            except Exception as e:
                print(f"Optical Flow Error: {e}")
                good_new, good_old = None, None

            if good_new is not None and len(good_new) > 0:
                self.old_points = good_new.reshape(-1, 1, 2)
            else:
                self.old_points = None
                good_new, good_old = None, None

        self.old_gray = frame_gray.copy()

        return features, good_new, good_old
