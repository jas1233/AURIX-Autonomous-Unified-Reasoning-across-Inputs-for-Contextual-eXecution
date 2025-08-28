# AURIX (Autonomous Unified Reasoning across Inputs for Contextual eXecution)- Detailed Guide
# Complete installation and configuration instructions

## 🎯 Project Overview

This Smart AI Monitoring System implements 4 parallel processing components:

1. **Camera + Object Detection**: Real-time YOLO object detection with OCR, analyzed by GPT-4 mini
2. **Screen Monitoring**: Automated screen capture with OCR and AI recommendations every 2 seconds  
3. **Voice Commands**: Hindi/English speech recognition with system command execution
4. **Call Monitoring**: Suspicious keyword detection for spam/fraud prevention

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **Webcam** (built-in or USB)
- **Microphone** for voice commands
- **OpenAI API Key** ([Get one here](https://platform.openai.com/account/api-keys))

### 1. System Dependencies

#### Windows:
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Verify installation
tesseract --version
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-hin
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y libopencv-dev python3-opencv
```

#### macOS:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install tesseract tesseract-lang portaudio
```

### 2. Python Environment Setup

```bash
# Clone or download the project
git clone <your-repo-url>
cd smart-monitoring-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Configuration

1. **Copy configuration template:**
```bash
cp config.yaml.template config.yaml
```

2. **Edit config.yaml:**
```yaml
# Add your OpenAI API key
openai:
  api_key: "your-actual-api-key-here"

# Set correct Tesseract path
ocr:
  tesseract_path: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Windows
  # tesseract_path: "/usr/bin/tesseract"                              # Linux
  # tesseract_path: "/opt/homebrew/bin/tesseract"                     # macOS
```

3. **Test installations:**
```bash
python test_setup.py
```

## 🔧 Detailed Installation

### YOLO Model Setup

The system will automatically download YOLOv8 models on first run. For manual setup:

```bash
# Download specific models (optional)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### Audio System Setup

#### Windows Audio Issues:
```bash
# If PyAudio installation fails:
pip install pipwin
pipwin install pyaudio

# Alternative: Install from wheel
pip install https://www.lfd.uci.edu/~gohlke/pythonlibs/PyAudio-0.2.11-cp39-cp39-win_amd64.whl
```

#### Linux Audio Setup:
```bash
# Install ALSA development files
sudo apt install -y libasound2-dev

# Test microphone
arecord -l
```

#### macOS Audio Setup:
```bash
# Grant microphone permissions
# System Preferences > Security & Privacy > Privacy > Microphone
# Add Terminal and Python to allowed apps
```

### GPU Acceleration (Optional)

For better performance with YOLO detection:

#### NVIDIA GPU:
```bash
# Install CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎮 Usage Instructions

### Basic Usage

1. **Start the system:**
```bash
python smart-monitoring-system.py
```

2. **Web Interface (Optional):**
```bash
# Start web dashboard
python web_interface.py

# Open browser to: http://localhost:5000
```

### Voice Commands

The system supports both Hindi and English commands:

#### English Commands:
- "Turn on flashlight" / "Turn off flashlight"
- "Take screenshot"
- "Increase volume" / "Decrease volume"
- "Open calculator"
- "What's the weather?"
- "Set timer for 5 minutes"

#### Hindi Commands:
- "लाइट चालू करो" / "लाइट बंद करो"
- "स्क्रीनशॉट लो"
- "आवाज़ बढ़ाओ" / "आवाज़ कम करो"
- "कैलकुलेटर खोलो"
- "मौसम कैसा है?"

### System Controls

#### Master Controls:
```python
# Start all modules
system.start_system()

# Stop specific module
system.status['camera_active'] = False

# Get current results
results = system.get_system_results()

# Emergency shutdown
system.shutdown()
```

## 📊 Monitoring & Logs

### Log Files:
- `monitoring_system.log`: Main system log
- `errors.log`: Error tracking
- `results.log`: Detection and analysis results

### Performance Monitoring:
```bash
# View real-time stats
tail -f monitoring_system.log

# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

### Results Export:
```bash
# Export results to CSV
python export_results.py --format csv --date today

# Export to JSON
python export_results.py --format json --hours 24
```

## 🔧 Troubleshooting

### Common Issues:

#### 1. Camera Not Working:
```bash
# Test camera access
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('Camera OK' if ret else 'Camera Failed')
cap.release()
"

# Check camera permissions (Windows/macOS)
# Settings > Privacy > Camera
```

#### 2. OCR Not Working:
```bash
# Test Tesseract
tesseract --version

# Test OCR with sample image
python -c "
import pytesseract
from PIL import Image
print(pytesseract.image_to_string(Image.new('RGB', (100, 100), 'white')))
"
```

#### 3. Speech Recognition Issues:
```bash
# Test microphone
python -c "
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print('Microphone detected')
"

# Check microphone permissions
```

#### 4. OpenAI API Issues:
```bash
# Test API connection
python -c "
import openai
openai.api_key = 'your-key'
try:
    response = openai.Completion.create(engine='gpt-3.5-turbo-instruct', prompt='Hello', max_tokens=5)
    print('API OK')
except Exception as e:
    print(f'API Error: {e}')
"
```

### Performance Optimization:

#### 1. Reduce CPU Usage:
```yaml
# In config.yaml
camera:
  detection_interval: 5  # Process every 5th frame instead of 3
  fps: 15               # Reduce camera FPS

screen:
  capture_interval: 5   # Capture every 5 seconds instead of 2
  resize_factor: 0.3    # Smaller resize factor
```

#### 2. Memory Optimization:
```yaml
performance:
  max_results_in_memory: 50  # Reduce from 100
  queue_sizes:
    camera: 5               # Reduce queue sizes
    screen: 5
```

#### 3. Use Lighter Models:
```yaml
yolo:
  model_path: "yolov8n.pt"  # Nano model (fastest)
  confidence_threshold: 0.7  # Higher threshold = fewer detections
```

## 🔐 Security Considerations

### API Key Security:
```bash
# Use environment variables instead of config file
export OPENAI_API_KEY="your-key-here"

# Or use .env file (add to .gitignore)
echo "OPENAI_API_KEY=your-key-here" > .env
pip install python-dotenv
```

### Privacy Settings:
```yaml
# In config.yaml
storage:
  auto_save_interval: 0      # Disable auto-save
  max_results_in_memory: 10  # Keep minimal data

logging:
  level: "ERROR"             # Reduce logging detail
```

## 📈 Advanced Configuration

### Custom YOLO Model:
```python
# Train custom model for specific objects
# Place custom .pt file in project directory
# Update config.yaml:
yolo:
  model_path: "custom_model.pt"
```

### Multi-Camera Setup:
```yaml
# Support multiple cameras
camera:
  devices: [0, 1, 2]  # Multiple camera indices
```

### Remote Monitoring:
```yaml
# Enable web API
api:
  enabled: true
  host: "0.0.0.0"  # Allow external connections
  port: 5000
  api_key_required: true
```

## 🆘 Support & Development

### Development Mode:
```bash
# Enable debug logging
python smart-monitoring-system.py --debug

# Use mock data when hardware unavailable
python smart-monitoring-system.py --mock
```

### Testing:
```bash
# Run unit tests
python -m pytest tests/

# Test individual components
python test_camera.py
python test_speech.py
python test_ocr.py
```

### Contributing:
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request

### Known Limitations:
- Call monitoring requires additional hardware integration
- Some voice commands may need system-specific implementation
- GPU acceleration improves performance but not required
- Real-time processing depends on hardware capabilities

### Performance Benchmarks:
- **Minimum**: Intel i5, 8GB RAM, integrated graphics
- **Recommended**: Intel i7/AMD Ryzen 7, 16GB RAM, dedicated GPU
- **Optimal**: High-end CPU, 32GB RAM, RTX 3060+ or equivalent

## 📝 License & Credits

This project uses several open-source libraries:
- **OpenCV** for computer vision
- **YOLO (Ultralytics)** for object detection  
- **Tesseract** for OCR
- **SpeechRecognition** for voice processing
- **OpenAI** for AI analysis

See `LICENSE.md` for full license information.
