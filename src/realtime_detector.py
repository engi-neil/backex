import os
import sys
import time
import cv2
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow_core import OpticalFlowTracker

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ==============================================================================
# CONFIGURATION AND PARAMETERS
# ==============================================================================

# Camera resolution
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Sequence properties
SEQ_LEN = 60
FEATURE_DIM = 8

# Shi-Tomasi keypoint detection parameters
FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# ==============================================================================
# HARDWARE / CAMERA ABSTRACTION
# ==============================================================================

class CameraStream:
    """Handles camera connection with Picamera2 (preferred) or cv2.VideoCapture fallback."""
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.use_picamera2 = False
        self.picam2 = None
        self.cap = None

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "XRGB8888"
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.picam2.set_controls({
                "AwbEnable": True,
                "ColourGains": (2.0, 2.0)
})
            self.use_picamera2 = True
            print("Successfully initialized Picamera2.")
        except ImportError:
            print("Picamera2 not found. Falling back to cv2.VideoCapture(0).")
            self._init_cv2()
        except Exception as e:
            print(f"Failed to initialize Picamera2: {e}. Falling back to cv2.VideoCapture(0).")
            self._init_cv2()

    def _init_cv2(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera via OpenCV.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Returns (ret, frame). Frame is in BGR format for OpenCV compatibility."""
        if self.use_picamera2:
            try:
                frame_rgb = self.picam2.capture_array('main')
                # Picamera2's "XRGB8888" format returns arrays already ordered (B, G, R, X)
                # in memory -- just drop the alpha channel, don't re-swap R/B.
                # FIX: Add .copy() to ensure a contiguous memory allocation
                frame_bgr = frame_rgb[:, :, :3].copy()
                return True, frame_bgr
            except Exception as e:
                print(f"Picamera2 capture error: {e}")
                return False, None
        else:
            ret, frame = self.cap.read()
            if ret and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            return ret, frame

    def release(self):
        if self.use_picamera2:
            self.picam2.stop()
        else:
            if self.cap:
                self.cap.release()

# ==============================================================================
# OPTICAL FLOW FEATURE EXTRACTOR
# ==============================================================================

class OpticalFlowExtractor(OpticalFlowTracker):
    """Shi-Tomasi + Lucas-Kanade point tracking, constrained to an ROI to
    avoid tracking stray points on background clutter. Thin wrapper around
    the shared OpticalFlowTracker (src/optical_flow_core.py) so the live
    inference pipeline and the offline training feature extractor
    (extract_features.py) cannot silently diverge -- see git history for the
    bug this caused when the two were separate implementations."""
    def __init__(self):
        super().__init__(feature_params=FEATURE_PARAMS, lk_params=LK_PARAMS)


# ==============================================================================
# MAIN DETECTOR APPLICATION
# ==============================================================================

class RealtimeDetector:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        
        # Dependencies paths
        self.model_path = self.model_dir / "gru_model.keras"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.ex_enc_path = self.model_dir / "exercise_encoder.pkl"
        self.form_enc_path = self.model_dir / "form_encoder.pkl"

        # Hardware/Model variables
        self.camera = None
        self.model = None
        self.scaler = None
        self.exercise_encoder = None
        self.form_encoder = None

        # Data buffers
        self.feature_buffer = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        self.frame_count = 0

        # Output states
        self.current_exercise = "Waiting..."
        self.current_form = "Waiting..."
        self.exercise_conf = 0.0
        self.form_conf = 0.0
        self.exec_speed = "Normal"
        
        self.extractor = OpticalFlowExtractor()

    def _load_assets(self):
        """Loads TF model, Scaler, and Encoders."""
        try:
            print("Loading GRU model...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            print("Loading preprocessing objects...")
            self.scaler = joblib.load(self.scaler_path)
            self.exercise_encoder = joblib.load(self.ex_enc_path)
            self.form_encoder = joblib.load(self.form_enc_path)
            
            # Warmup the model with dummy data
            dummy_input = np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
            self.model(dummy_input, training=False)
            print("Assets loaded and model warmed up successfully.")
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure gru_model.keras, scaler.pkl, exercise_encoder.pkl, and form_encoder.pkl exist in the 'models' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during initialization: {e}")
            sys.exit(1)

    def _classify_speed(self, mean_magnitude):
        """Classifies execution speed based on optical flow magnitude."""
        if mean_magnitude < 0.5:
            return "Slow"
        elif mean_magnitude > 4.0:
            return "Fast"
        return "Normal"

    def _draw_overlay(self, frame, good_new, good_old, fps):
        """Renders tracking visualization and predictions onto the frame."""
        # Visualization: Draw optical flow tracks
        if good_new is not None and good_old is not None:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw motion vector line
                frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), COLOR_YELLOW, 2)
                # Draw feature point
                frame = cv2.circle(frame, (int(a), int(b)), 3, COLOR_BLUE, -1)

            # Optional Bounding Box based on extreme flow points
            if len(good_new) > 4:
                x, y, w, h = cv2.boundingRect(np.array(good_new))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
        
        # Color conditional on form
        form_color = COLOR_GREEN if self.current_form.lower() == "good" else COLOR_RED
        if self.current_form == "Waiting...":
            form_color = COLOR_WHITE

        # Overlay text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Rendering texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        cv2.putText(frame, f"Exercise: {self.current_exercise} ({self.exercise_conf:.2f})", (5, 15), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"Form: {self.current_form} ({self.form_conf:.2f})", (5, 30), font, font_scale, form_color, thickness)
        cv2.putText(frame, f"Speed: {self.exec_speed}", (5, 45), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"Tracked Pts: {len(good_new) if good_new is not None else 0}", (5, 60), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (5, 75), font, font_scale, COLOR_WHITE, thickness)
        
        # Progress bar for buffer
        if self.frame_count < SEQ_LEN:
            progress_pct = self.frame_count / SEQ_LEN
            cv2.putText(frame, "Buffering...", (5, 90), font, font_scale, COLOR_YELLOW, thickness)
            cv2.rectangle(frame, (80, 82), (80 + int(100 * progress_pct), 92), COLOR_YELLOW, -1)
            cv2.rectangle(frame, (80, 82), (180, 92), COLOR_WHITE, 1)

        return frame

    def run(self):
        """Main execution loop."""
        self._load_assets()
        
        print("Starting camera stream...")
        try:
            self.camera = CameraStream()
        except RuntimeError as e:
            print(e)
            sys.exit(1)

        cv2.namedWindow("Realtime Exercise Form Monitor", cv2.WINDOW_AUTOSIZE)
        
        prev_time = time.perf_counter()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Camera disconnected or frame dropped. Retrying...")
                    time.sleep(0.5)
                    continue

                curr_time = time.perf_counter()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Feature Extraction
                features, good_new, good_old = self.extractor.process_frame(frame_gray)

                # 2. Buffer Management (Shift left, insert new at end)
                self.feature_buffer[:-1] = self.feature_buffer[1:]
                self.feature_buffer[-1] = features
                self.frame_count = min(self.frame_count + 1, SEQ_LEN)

                # Update speed classification based on recent magnitude
                self.exec_speed = self._classify_speed(features[2])

                # 3. Model Inference (only if buffer is full)
                if self.frame_count >= SEQ_LEN and self.frame_count % 3 == 0:
                    # StandardScaler transform
                    # Reshape to 2D for sklearn scaler, then back to 3D for GRU
                    buffer_2d = self.feature_buffer.reshape(-1, FEATURE_DIM)
                    scaled_buffer = self.scaler.transform(buffer_2d).astype(np.float32)
                    model_input = scaled_buffer.reshape(1, SEQ_LEN, FEATURE_DIM)
                    
                    # Predict (Using __call__ for performance on single samples)
                    preds = self.model(model_input, training=False)
                    exercise_probs, form_probs = preds[0].numpy(), preds[1].numpy()
                    
                    # Extract predictions
                    ex_idx = np.argmax(exercise_probs, axis=1)[0]
                    form_idx = np.argmax(form_probs, axis=1)[0]
                    
                    self.current_exercise = self.exercise_encoder.inverse_transform([ex_idx])[0]
                    self.current_form = self.form_encoder.inverse_transform([form_idx])[0]
                    
                    self.exercise_conf = exercise_probs[0][ex_idx]
                    self.form_conf = form_probs[0][form_idx]

                # 4. Display
                display_frame = self._draw_overlay(frame, good_new, good_old, fps)
                cv2.imshow("Realtime Exercise Form Monitor", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Shutdown requested by user.")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down...")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()


def main():
    # Setup TensorFlow optimizations for RPi
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    BASE_DIR = Path(__file__).resolve().parent.parent / "back_exercise_form_detector"
    MODEL_DIR = BASE_DIR / "models"

    detector = RealtimeDetector(model_dir=MODEL_DIR)
    detector.run()

if __name__ == "__main__":
    main()
