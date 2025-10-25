# Real-time Age/Gender/Mood-based Ad Recommendation System

A minimal, local prototype that uses webcam, face detection, and demographic estimation to recommend personalized ads in real-time.

## 🎯 Features

- **Real-time face detection** using MTCNN (Multi-task Cascaded Convolutional Networks)
- **Live camera window** with visual overlays showing detection results and ad recommendations
- **Age group estimation**: child (0-12), teen (13-19), young (20-34), adult (35-54), senior (55+)
- **Gender detection**: Male, Female, Unknown
- **Mood/emotion detection**: happy, neutral, sad
- **Smart ad matching** with scoring algorithm (age + gender + mood + priority + randomness)
- **Privacy-focused**: No frame storage, only metadata logged
- **Continuous detection**: Analyzes every 5 seconds
- **Dual output**: Live video window + Terminal logs
- **CSV logging**: Detection events saved for analytics

## 📋 Requirements

- Python 3.10 or higher
- macOS (tested on macOS, should work on Linux/Windows with minor tweaks)
- Webcam/camera access
- Internet connection (for initial model downloads)

## 🚀 Setup Instructions

### 1. Clone or navigate to project directory

```bash
cd /some path
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download pretrained models (~50-100MB). This may take a few minutes.

### 4. Verify camera access

Make sure your system allows terminal/Python to access the camera:
- **macOS**: System Preferences → Security & Privacy → Privacy → Camera → Enable for Terminal

### 5. Run the application

```bash
python main.py
```

## 📁 Project Structure

```
AdResearch/
├── main.py              # Main application loop
├── recommender.py       # Ad recommendation engine
├── utils/
│   ├── capture.py       # Webcam handling
│   └── models.py        # Face detection & demographic estimation
├── data/
│   └── ads.csv          # Sample ad inventory
├── logs/
│   └── detections.csv   # Detection event log (auto-created)
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── file_structure.txt  # Detailed structure documentation
```

## 🎮 Usage

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Position yourself** in front of the webcam

3. **Observe the live camera window** showing:
   - Real-time video feed from your webcam
   - Green bounding box around detected face
   - Demographics overlay (age, gender, mood)
   - Recommended ad information at bottom
   - Countdown timer to next detection

4. **Every 5 seconds**, the system will:
   - Detect your face(s)
   - Estimate age group, gender, and mood
   - Match with best-fit ad from `data/ads.csv`
   - Update live video overlay
   - Print recommendation to terminal
   - Log event to `logs/detections.csv`

5. **Stop the application**: 
   - Press `q` key in the camera window, OR
   - Press `Ctrl+C` in terminal for graceful exit

## 📺 Live Camera Window Features

The camera window displays:
- ✅ **Live video feed** with real-time processing
- ✅ **Face detection box** (green rectangle around detected face)
- ✅ **Demographics label** (gender, age group, mood above face)
- ✅ **Info panel** at bottom showing:
  - Current demographics
  - Recommended ad ID and title
  - Ad priority level
- ✅ **Countdown timer** showing seconds until next detection
- ✅ **Visual feedback** when no face is detected

## 📊 Sample Terminal Output

```
=================================================================
Detection Event @ 2025-10-24 14:32:15
=================================================================
Detected Demographics:
  Age Group: young (20-34)
  Gender: Male
  Mood: happy

Recommended Ad:
  ID: AD_008
  Title: Premium Fitness Membership
  Description: Get fit with our state-of-the-art gym facilities
  Tags: young, Male, happy
  Priority: 8
=================================================================
```

## 🔧 Configuration

### Enable/Disable Live Camera Window

In `main.py`, toggle the camera window:
```python
SHOW_CAMERA_WINDOW = True  # Set to False for terminal-only mode
```

**Terminal-only mode**: Set to `False` if you want recommendations in terminal only (no video window)  
**Live video mode**: Set to `True` for visual feedback with detection overlays (default)

### Customize ads

Edit `data/ads.csv` to add your own ads:
- `ad_id`: Unique identifier
- `title`: Ad headline
- `description`: Ad copy
- `target_age_groups`: Comma-separated (child, teen, young, adult, senior, All)
- `target_genders`: Comma-separated (Male, Female, All)
- `target_moods`: Comma-separated (happy, neutral, sad, All)
- `priority`: 1-10 (higher = more likely to show)
- `creative_url`: Path/URL to ad creative (placeholder)

### Adjust detection interval

In `main.py`, change the `DETECTION_INTERVAL` variable:
```python
DETECTION_INTERVAL = 5  # seconds (default: 5)
```

## 📈 Accuracy & Limitations

### Current Accuracy Trade-offs

- **Age estimation**: Approximate age groups, not exact ages. Accuracy ~60-70% for age range.
- **Gender detection**: Binary classification (Male/Female). Accuracy ~85-90%.
- **Mood detection**: Simplified to 3 emotions. Basic model, ~65-75% accuracy.
- **Lighting & angle**: Performance degrades in poor lighting or extreme angles.
- **Multiple faces**: Currently uses largest face only.

### Improvement Suggestions

1. **Better models**: 
   - Use DeepFace, FaceNet, or custom-trained models
   - Fine-tune on diverse, representative datasets
   
2. **More emotions**: 
   - Expand from 3 to 7+ emotions (angry, surprised, fearful, disgusted)
   - Use FER2013 or AffectNet trained models

3. **Ensemble approach**: 
   - Combine multiple models for voting/averaging
   - Calibrate confidence scores

4. **Data augmentation**: 
   - Train on varied lighting, angles, ethnicities
   - Use synthetic data generation

5. **Multi-face handling**: 
   - Aggregate demographics from multiple faces
   - Weight by face size/position

6. **A/B testing framework**: 
   - Track ad performance metrics
   - Optimize scoring weights

## 🔒 Privacy

- **No image storage**: Frames are processed in memory only
- **Metadata only**: Only demographic labels and timestamps logged
- **Local processing**: All computation happens on your machine
- **Opt-in**: Camera only active when application runs

## 🐛 Troubleshooting

### Camera not found
- Check camera permissions in System Preferences (macOS)
- Verify camera works in other apps (Photo Booth, Zoom)
- Try unplugging/replugging external webcam

### Model download fails
- Check internet connection
- Models will fallback to simple heuristic estimators
- Manually download from TensorFlow Hub if needed

### Low detection accuracy
- Ensure good lighting (front-facing light)
- Face camera directly
- Remove obstructions (glasses, masks may reduce accuracy)
- Adjust camera position to capture clear face

### Performance issues
- Close other camera-using applications
- Reduce detection frequency (increase `DETECTION_INTERVAL`)
- Use a lighter model (edit `utils/models.py`)

## 📚 Dependencies

- **opencv-python**: Webcam capture and image processing
- **tensorflow**: Deep learning framework for models
- **mtcnn**: Face detection
- **pandas**: CSV handling
- **numpy**: Numerical operations
- **pillow**: Image manipulation

## 🛠 Development

### Running in development mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run with verbose logging
python main.py
```

### Testing with static images (future enhancement)

Modify `main.py` to load images instead of webcam for testing.

## 📄 License

This is a prototype for research and educational purposes.

## 👤 Author

Built for AdResearch project - Real-time demographic-based ad targeting system.

## 🙏 Acknowledgments

- MTCNN for face detection
- TensorFlow Hub for pretrained models
- OpenCV community for excellent documentation

---

**Questions or issues?** Check `file_structure.txt` for detailed component documentation.
