# AdResearch Project — Documentation

Last updated: 2025-10-25

This document describes the AdResearch project: what it does, the technologies used and how they are used, the runtime workflow, outputs, and step-by-step setup instructions to get a new developer up and running.

## 1. Project Overview

AdResearch is a local prototype that performs real-time face detection from a webcam and recommends advertisements based on simple demographic estimations (age group, gender, mood). The ad inventory is stored locally as a CSV file. The prototype is intended for research and experimentation — it is NOT intended for production use without careful ethical review and improved models.

Key goals:
- Demonstrate real-time face detection using MTCNN
- Provide simple age/gender/mood estimation for ad targeting experiments
- Score and recommend ads from a CSV inventory
- Provide an optional live camera window showing detections and overlays
- Log detection events for later analysis

## 2. High-level Architecture

- `main.py` — application entry point. Captures frames from webcam, invokes detection, prediction and recommendation, and handles UI (terminal + optional OpenCV window) and logging.
- `utils/capture.py` — webcam management and safe capture helper.
- `utils/models.py` — `FaceEstimator` class; loads MTCNN detector and contains demographic estimators (current prototype uses heuristic estimators; comments explain where to plug in TensorFlow models).
- `recommender.py` — scoring engine that matches detected demographics to ads and ranks them.
- `data/ads.csv` — ad inventory file (CSV) used by recommender.
- `logs/detections.csv` — runtime event log (detection timestamps, demographics, recommended ad).
- `requirements.txt` — Python dependencies used by the project.

## 3. Technologies and How They Are Used

- Python 3.9 (recommended) — main language for the prototype.
- OpenCV (`cv2`) — camera capture, image conversion (BGR/RGB), and optional live display with overlays.
- MTCNN (via the `mtcnn` package) — face detection. MTCNN internally depends on TensorFlow/Keras. It detects bounding boxes and facial landmarks (eyes, nose, mouth).
- TensorFlow (`tensorflow-macos` on macOS) — required by MTCNN. The current prototype uses heuristic methods for age/gender/mood prediction rather than training/using dedicated TF models, but TF is required for the MTCNN detector itself.
- NumPy — image arrays and numeric operations.
- Pandas — reading/writing CSV ad inventory and logs (if used by the recommender or utilities).
- Pillow (PIL) — image manipulation (if/when used in utilities).

How each technology is used in this project:
- Camera capture (OpenCV): `utils/capture.py` opens the default camera, reads frames and hands frames to `main.py`.
- Face detection (MTCNN): `utils/models.py` constructs an `MTCNN()` detector and calls `detect_faces(rgb_frame)` to obtain face boxes and keypoints.
- Demographic estimation (heuristics for prototype): After cropping a face, `FaceEstimator._extract_face_features()` computes brightness and texture (Laplacian variance). `predict_age()`, `predict_gender()`, and `predict_mood()` use those features combined with small weighted randomness to return labels. The code is intentionally lightweight and annotated where to plug in trained TensorFlow/Keras models.
- Recommender scoring: `recommender.py` loads `data/ads.csv`, computes a score per ad for detected demographics, and returns the top candidate(s).
- Logging: Each detection + chosen ad is appended to `logs/detections.csv` for offline analysis.

## 4. Workflow (Runtime)

1. Start the application (`main.py`).
2. The webcam is opened and a capture loop begins.
3. Every N seconds (configurable via `DETECTION_INTERVAL`), the current frame is captured and passed to `FaceEstimator.detect_faces()`.
4. For each detected face: crop the face image, then call `predict_age()`, `predict_gender()`, and `predict_mood()`.
5. The `AdRecommender` scores each ad in `data/ads.csv` against the detected demographics and picks the highest scoring ad.
6. Output:
   - The recommended ad is printed to the terminal (title, description, creative URL).
   - If `SHOW_CAMERA_WINDOW` is enabled, an OpenCV window displays the camera frame annotated with bounding boxes and demographics labels.
   - An entry is appended to `logs/detections.csv` with timestamp, demographics, and chosen ad metadata.
7. Repeat until the program is stopped.

## 5. Expected Outputs

- Terminal: A readable summary of recommended ad(s) each detection interval.
- Optional OpenCV window: Live camera view with rectangles around faces and labels (age group, gender, mood) plus recommended ad overlay.
- `logs/detections.csv`: CSV log entries with one row per detection event including timestamp, age_group, gender, mood, ad_id, ad_title, score, and optionally the face bounding box.

Example `logs/detections.csv` columns (may vary by implementation):
- timestamp, face_id, x, y, width, height, age_group, gender, mood, ad_id, ad_title, score

## 6. Where Demographic Predictions Come From (Important)

- Face detection: MTCNN provides bounding boxes and landmarks. MTCNN requires TensorFlow/Keras to run. It is already part of `requirements.txt` as `tensorflow-macos` (macOS) or `tensorflow` for other platforms.
- Age/Gender/Mood predictions in the current repository are implemented as heuristic estimators (brightness, texture, and weighted randomness). These are intentionally simple to keep the prototype small and to avoid shipping large additional pre-trained models.

If you want to replace the heuristics with trained models, you can:
- Train or obtain pre-trained TensorFlow/Keras models for age, gender and emotion (examples: SSR-Net, VGGFace2 fine-tuned, FER2013 models).
- Save models to `models/` (e.g., `models/age_model.h5`) and load them in `utils/models.py` in `_initialize_models()` where placeholders are currently documented.

Tradeoffs:
- Using trained models increases accuracy significantly but adds large model files and may require GPU/optimized builds for real-time performance.
- Heuristics are fast, small, and transparent but not accurate.

## 7. Security, Privacy and Ethics (Please read)

This project reads live video and infers sensitive attributes (age, gender, mood). Before using this against people, consider legal and ethical requirements:
- Obtain explicit consent from people being recorded.
- Anonymize or avoid storing personally identifying images.
- Avoid using demographic inferences for sensitive decisions.
- Follow local privacy laws and GDPR/CCPA where applicable.

## 8. Setup Instructions (macOS) — Quick Start

Prerequisites:
- Python 3.9 (or compatible 3.8–3.11); pyenv or system Python
- Homebrew (optional) for installing dependencies

Steps:

1. Clone or copy the repository to your machine (if not already present):

```bash
cd ~/some/path
git clone https://github.com/TehanIsum/Ad_Research_Project.git
cd Ad_Research_Project
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes for macOS users: `requirements.txt` contains `tensorflow-macos` when this project was assembled on macOS — that package is large (~800+ MB). If you do not want TensorFlow, you can switch to an alternative face detector (see Troubleshooting / Alternatives), but MTCNN requires TF.

4. Configure options (optional):
- `SHOW_CAMERA_WINDOW`: toggle whether to open an annotated camera window (edit `main.py` or pass as environment var if implemented).
- `DETECTION_INTERVAL`: seconds between detection cycles.

5. Run the app:

```bash
python main.py
```

You should see either terminal output recommending ads every detection interval, and/or a live camera window with annotated detections.

## 9. How to Add Ads

- Open `data/ads.csv` in a spreadsheet or text editor.
- Each row contains an ad with columns such as: ad_id, title, description, target_age_groups, target_genders, target_moods, priority, creative_url
- Example: `1, Cozy Coffee Ad, "Hot coffee for chilly mornings", young|adult, Female|Male, happy|neutral, 8, https://example.com/creative1.jpg`

The recommender parses the CSV and matches against the detected demographics. See `recommender.py` to customize the scoring logic.

## 10. Troubleshooting & Alternatives

- Problem: "MTCNN fails to import without TensorFlow"
  - Cause: the `mtcnn` package depends on `tensorflow.keras.*` internally. You must have TensorFlow/Keras available.
  - Fix: install `tensorflow-macos` (macOS) or `tensorflow` (Linux/Windows) in the virtual environment.

- Problem: "Too much disk space used by venv / TensorFlow is large"
  - Explanation: TensorFlow is large (~800MB) and common for ML projects. Options: use an alternative face detector that doesn't need TF (tradeoffs follow).

- Alternatives (no TensorFlow required):
  - OpenCV Haar Cascades (cv2.CascadeClassifier) — very small, lower accuracy, fast on CPU.
  - MediaPipe Face Detection — accurate and fast, minimal dependencies (Google's MediaPipe adds a binary dependency), can be GPU-accelerated.
  - Dlib's HOG or CNN detectors — dlib has binary dependencies and may be heavy to compile.

If you switch to an alternative detector, update `utils/models.py` to call the new detector and remove the MTCNN/TensorFlow dependency from `requirements.txt`.

## 11. Testing

- There is a simple camera test utility `test_camera_window.py` which opens the camera and displays frames with optional overlays. Use it to verify camera access.
- Unit tests are not currently included. Suggested minimal tests:
  - Test CSV parsing for `data/ads.csv`.
  - Test `FaceEstimator._extract_face_features()` on sample images.
  - Test recommender scoring with controlled demographic inputs.

## 12. Development & Contribution

- Fork the repository and submit pull requests for bug fixes and features.
- Add unit tests for any new logic and keep changes small and reviewable.
- If adding trained models, store them in `models/` and add an entry to `.gitignore` for large files; consider Git LFS for model files.

## 13. File Map (common files)

- `main.py` — main application loop
- `recommender.py` — ad matching / scoring engine
- `utils/models.py` — MTCNN wrapper and demographic estimators (heuristics)
- `utils/capture.py` — webcam helper
- `data/ads.csv` — sample ad inventory
- `logs/detections.csv` — detection log (created at runtime)
- `requirements.txt` — Python dependencies
- `.gitignore` — repository ignore rules

## 14. Next steps & optional improvements

- Replace heuristic estimators with pre-trained TF/Keras models for age/gender/emotion.
- Add unit tests and CI workflow that runs tests on push.
- Add an admin UI to manage ads and visualize logs.
- Add privacy-preserving features: on-device anonymization, image hashing instead of saving raw images.

## 15. Contact & Acknowledgements

Author / Maintainer: TehanIsum

Acknowledgements: MTCNN implementation (https://github.com/ipazc/mtcnn), OpenCV, TensorFlow

---

If you want, I can also:
- Commit and push this `DOCUMENTATION.md` to your remote branch (I already helped push other files earlier). Tell me how you'd like to authenticate (SSH or Personal Access Token), or I can prepare a PR/branch for manual review.
- Expand any specific section into a dedicated README subsection or add diagrams.
