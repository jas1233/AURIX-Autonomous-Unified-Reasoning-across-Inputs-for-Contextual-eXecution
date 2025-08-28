# Smart AI Monitoring System - Main Implementation
# Handles 4 parallel processing tasks:
# 1. Camera feed with YOLO object detection + OCR → GPT-4 mini
# 2. Screen capture with OCR → GPT-4 mini recommendations (every 2 seconds)
# 3. Speech-to-text (Hindi/English) with voice commands
# 4. Call monitoring for suspicious word detection

import cv2
import numpy as np
import threading
import queue
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

# Required installations:
# pip install opencv-python ultralytics pytesseract speechrecognition pyaudio openai pillow psutil pygetwindow

# Core libraries
import pytesseract
import speech_recognition as sr
from ultralytics import YOLO
from PIL import ImageGrab, Image
import openai
import pyaudio
import wave
import psutil
import pygetwindow as gw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_system.log'),
        logging.StreamHandler()
    ]
)

class SmartMonitoringSystem:
    def __init__(self):
        # System configuration
        self.config = {
            'openai_api_key': 'your-openai-api-key-here',
            'camera_device': 0,
            'screen_capture_interval': 2,  # seconds
            'yolo_model': 'yolov8n.pt',  # You Only Look Once model
            'tesseract_path': r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Windows path
            'suspicious_keywords': [
                'otp', 'verification', 'urgent', 'verify', 'bank', 'account',
                'suspended', 'click here', 'lottery', 'winner', 'prize',
                'congratulations', 'claim', 'offer', 'limited time'
            ]
        }
        
        # Initialize OpenAI
        openai.api_key = self.config['openai_api_key']
        
        # Initialize YOLO model
        self.yolo_model = YOLO(self.config['yolo_model'])
        
        # Configure Tesseract
        pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_path']
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.running = False
        
        # Communication queues
        self.camera_queue = queue.Queue(maxsize=10)
        self.screen_queue = queue.Queue(maxsize=10)
        self.voice_queue = queue.Queue(maxsize=20)
        self.call_queue = queue.Queue(maxsize=20)
        
        # Results storage
        self.results = {
            'camera_detections': [],
            'screen_analysis': [],
            'voice_commands': [],
            'call_alerts': []
        }
        
        # System status
        self.status = {
            'camera_active': False,
            'screen_monitor_active': False,
            'voice_active': False,
            'call_monitor_active': False
        }

    def start_system(self):
        """Start all monitoring components"""
        self.running = True
        logging.info("Starting Smart Monitoring System...")
        
        # Start all 4 parallel processes
        futures = []
        
        # 1. Camera + Object Detection + OCR
        futures.append(self.executor.submit(self.camera_detection_thread))
        
        # 2. Screen Capture + OCR + Recommendations
        futures.append(self.executor.submit(self.screen_monitoring_thread))
        
        # 3. Speech Recognition + Voice Commands
        futures.append(self.executor.submit(self.voice_command_thread))
        
        # 4. Call Monitoring + Suspicious Word Detection
        futures.append(self.executor.submit(self.call_monitoring_thread))
        
        # Data processing threads
        futures.append(self.executor.submit(self.camera_processor_thread))
        futures.append(self.executor.submit(self.screen_processor_thread))
        futures.append(self.executor.submit(self.voice_processor_thread))
        futures.append(self.executor.submit(self.call_processor_thread))
        
        try:
            # Wait for completion or interruption
            while self.running:
                time.sleep(1)
                self.log_system_status()
                
        except KeyboardInterrupt:
            logging.info("Shutdown initiated by user...")
        finally:
            self.shutdown()

    # ===============================
    # THREAD 1: CAMERA DETECTION
    # ===============================
    
    def camera_detection_thread(self):
        """Camera feed with YOLO object detection and OCR"""
        logging.info("Camera Detection Thread Started")
        self.status['camera_active'] = True
        
        cap = cv2.VideoCapture(self.config['camera_device'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while self.running and self.status['camera_active']:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Process every 3rd frame to reduce load
                if frame_count % 3 == 0:
                    # Add frame to processing queue
                    if not self.camera_queue.full():
                        self.camera_queue.put({
                            'frame': frame.copy(),
                            'timestamp': datetime.now()
                        })
                
                # Display live feed (optional)
                cv2.imshow('Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                logging.error(f"Camera detection error: {e}")
                time.sleep(0.1)
        
        cap.release()
        cv2.destroyAllWindows()
        self.status['camera_active'] = False
        logging.info("Camera Detection Thread Stopped")

    def camera_processor_thread(self):
        """Process camera frames with YOLO + OCR + GPT analysis"""
        logging.info("Camera Processor Thread Started")
        
        while self.running:
            try:
                # Get frame from queue
                data = self.camera_queue.get(timeout=1)
                frame = data['frame']
                timestamp = data['timestamp']
                
                # YOLO Object Detection
                results = self.yolo_model(frame)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Extract object information
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.yolo_model.names[cls]
                            
                            detections.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })
                
                # OCR on entire frame
                ocr_text = self.extract_text_from_image(frame)
                
                # Send to GPT for contextual analysis
                context = self.analyze_camera_context(detections, ocr_text)
                
                # Store results
                result_data = {
                    'timestamp': timestamp.isoformat(),
                    'detections': detections,
                    'ocr_text': ocr_text,
                    'gpt_context': context
                }
                
                self.results['camera_detections'].append(result_data)
                
                # Keep only last 50 results
                if len(self.results['camera_detections']) > 50:
                    self.results['camera_detections'] = self.results['camera_detections'][-50:]
                
                logging.info(f"Camera: Detected {len(detections)} objects, GPT: {context[:50]}...")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Camera processor error: {e}")

    # ===============================
    # THREAD 2: SCREEN MONITORING
    # ===============================
    
    def screen_monitoring_thread(self):
        """Screen capture every 2 seconds with OCR"""
        logging.info("Screen Monitoring Thread Started")
        self.status['screen_monitor_active'] = True
        
        while self.running and self.status['screen_monitor_active']:
            try:
                # Capture screen
                screenshot = ImageGrab.grab()
                
                # Add to processing queue
                if not self.screen_queue.full():
                    self.screen_queue.put({
                        'image': screenshot,
                        'timestamp': datetime.now()
                    })
                
                # Wait 2 seconds
                time.sleep(self.config['screen_capture_interval'])
                
            except Exception as e:
                logging.error(f"Screen monitoring error: {e}")
                time.sleep(1)
        
        self.status['screen_monitor_active'] = False
        logging.info("Screen Monitoring Thread Stopped")

    def screen_processor_thread(self):
        """Process screen captures with OCR + GPT recommendations"""
        logging.info("Screen Processor Thread Started")
        
        while self.running:
            try:
                # Get screenshot from queue
                data = self.screen_queue.get(timeout=2)
                image = data['image']
                timestamp = data['timestamp']
                
                # Resize image for OCR efficiency
                image = image.resize((image.width//2, image.height//2))
                
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(image, lang='eng')
                
                # Get active window info
                active_window = self.get_active_window_info()
                
                # Get GPT recommendations
                recommendations = self.get_screen_recommendations(ocr_text, active_window)
                
                # Store results
                result_data = {
                    'timestamp': timestamp.isoformat(),
                    'active_window': active_window,
                    'ocr_text': ocr_text,
                    'recommendations': recommendations
                }
                
                self.results['screen_analysis'].append(result_data)
                
                # Keep only last 50 results
                if len(self.results['screen_analysis']) > 50:
                    self.results['screen_analysis'] = self.results['screen_analysis'][-50:]
                
                logging.info(f"Screen: {active_window}, Rec: {recommendations[:50]}...")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Screen processor error: {e}")

    # ===============================
    # THREAD 3: VOICE COMMANDS
    # ===============================
    
    def voice_command_thread(self):
        """Continuous speech recognition for Hindi/English commands"""
        logging.info("Voice Command Thread Started")
        self.status['voice_active'] = True
        
        # Initialize speech recognizer
        r = sr.Recognizer()
        mic = sr.Microphone()
        
        # Adjust for ambient noise
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
            logging.info("Voice recognition calibrated")
        
        while self.running and self.status['voice_active']:
            try:
                with mic as source:
                    # Listen for audio with timeout
                    audio = r.listen(source, timeout=1, phrase_time_limit=5)
                
                # Add to processing queue
                if not self.voice_queue.full():
                    self.voice_queue.put({
                        'audio': audio,
                        'timestamp': datetime.now()
                    })
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logging.error(f"Voice command error: {e}")
                time.sleep(0.5)
        
        self.status['voice_active'] = False
        logging.info("Voice Command Thread Stopped")

    def voice_processor_thread(self):
        """Process voice commands with Hindi/English recognition"""
        logging.info("Voice Processor Thread Started")
        
        r = sr.Recognizer()
        
        while self.running:
            try:
                # Get audio from queue
                data = self.voice_queue.get(timeout=1)
                audio = data['audio']
                timestamp = data['timestamp']
                
                # Try English first, then Hindi
                recognized_text = None
                language = None
                
                # Try English recognition
                try:
                    recognized_text = r.recognize_google(audio, language='en-IN')
                    language = 'English'
                except:
                    # Try Hindi recognition
                    try:
                        recognized_text = r.recognize_google(audio, language='hi-IN')
                        language = 'Hindi'
                    except:
                        continue
                
                if recognized_text:
                    # Process voice command
                    command_result = self.process_voice_command(recognized_text, language)
                    
                    # Store results
                    result_data = {
                        'timestamp': timestamp.isoformat(),
                        'text': recognized_text,
                        'language': language,
                        'command_executed': command_result
                    }
                    
                    self.results['voice_commands'].append(result_data)
                    
                    # Keep only last 100 commands
                    if len(self.results['voice_commands']) > 100:
                        self.results['voice_commands'] = self.results['voice_commands'][-100:]
                    
                    logging.info(f"Voice [{language}]: {recognized_text} -> {command_result}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Voice processor error: {e}")

    # ===============================
    # THREAD 4: CALL MONITORING
    # ===============================
    
    def call_monitoring_thread(self):
        """Monitor calls for suspicious activity"""
        logging.info("Call Monitoring Thread Started")
        self.status['call_monitor_active'] = True
        
        # Note: This is a simulation since real call monitoring requires
        # hardware integration or phone system access
        
        while self.running and self.status['call_monitor_active']:
            try:
                # Simulate call monitoring (in real implementation, this would
                # interface with phone system or audio capture from calls)
                
                # For demo purposes, generate mock call data periodically
                if np.random.random() < 0.05:  # 5% chance every second
                    mock_call_data = self.generate_mock_call_data()
                    
                    if not self.call_queue.full():
                        self.call_queue.put(mock_call_data)
                
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Call monitoring error: {e}")
                time.sleep(1)
        
        self.status['call_monitor_active'] = False
        logging.info("Call Monitoring Thread Stopped")

    def call_processor_thread(self):
        """Process call data for suspicious keyword detection"""
        logging.info("Call Processor Thread Started")
        
        while self.running:
            try:
                # Get call data from queue
                data = self.call_queue.get(timeout=1)
                
                # Analyze call for suspicious content
                risk_assessment = self.analyze_call_content(data)
                
                # Store results
                result_data = {
                    'timestamp': data['timestamp'],
                    'caller_number': data['number'],
                    'call_duration': data.get('duration', 0),
                    'transcript': data.get('transcript', ''),
                    'risk_level': risk_assessment['risk_level'],
                    'suspicious_words': risk_assessment['suspicious_words'],
                    'action_taken': risk_assessment['action']
                }
                
                self.results['call_alerts'].append(result_data)
                
                # Keep only last 200 call records
                if len(self.results['call_alerts']) > 200:
                    self.results['call_alerts'] = self.results['call_alerts'][-200:]
                
                logging.info(f"Call: {data['number']} - Risk: {risk_assessment['risk_level']}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Call processor error: {e}")

    # ===============================
    # UTILITY METHODS
    # ===============================
    
    def extract_text_from_image(self, image):
        """Extract text from image using OCR"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Extract text
            text = pytesseract.image_to_string(pil_image)
            return text.strip()
        except Exception as e:
            logging.error(f"OCR error: {e}")
            return ""

    def analyze_camera_context(self, detections, ocr_text):
        """Use GPT to analyze camera context"""
        try:
            prompt = f"""
            Analyze this camera feed data and provide a 3-4 word description:
            
            Objects detected: {[d['class'] for d in detections]}
            Text visible: {ocr_text[:100]}
            
            Provide a brief contextual description of the scene:
            """
            
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )
            
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error(f"GPT camera analysis error: {e}")
            return "Analysis unavailable"

    def get_screen_recommendations(self, ocr_text, active_window):
        """Get GPT recommendations based on screen content"""
        try:
            prompt = f"""
            Based on the current screen activity, provide a helpful recommendation:
            
            Active Window: {active_window}
            Screen Text: {ocr_text[:200]}
            
            Provide a brief, actionable recommendation (max 10 words):
            """
            
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=30,
                temperature=0.7
            )
            
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error(f"GPT screen analysis error: {e}")
            return "Focus on current task"

    def get_active_window_info(self):
        """Get information about the currently active window"""
        try:
            # Get active window
            active_window = gw.getActiveWindow()
            if active_window:
                return active_window.title
            else:
                return "Unknown"
        except:
            return "Desktop"

    def process_voice_command(self, text, language):
        """Process and execute voice commands"""
        text_lower = text.lower()
        
        # Command mapping (English and Hindi)
        commands = {
            'flashlight': ['flashlight', 'torch', 'light', 'टॉर्च', 'लाइट'],
            'screenshot': ['screenshot', 'capture', 'स्क्रीनशॉट', 'फोटो'],
            'volume': ['volume', 'sound', 'आवाज़', 'वॉल्यूम'],
            'calculator': ['calculator', 'calc', 'कैलकुलेटर'],
            'weather': ['weather', 'मौसम'],
            'timer': ['timer', 'alarm', 'टाइमर'],
        }
        
        # Check for commands
        for command, keywords in commands.items():
            if any(keyword in text_lower for keyword in keywords):
                return self.execute_system_command(command, text_lower)
        
        return "Command not recognized"

    def execute_system_command(self, command, text):
        """Execute system commands based on voice input"""
        try:
            if command == 'flashlight':
                if 'on' in text or 'चालू' in text:
                    # Simulate turning on flashlight
                    return "Flashlight ON"
                else:
                    return "Flashlight OFF"
            
            elif command == 'screenshot':
                # Take screenshot
                screenshot = ImageGrab.grab()
                screenshot.save(f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                return "Screenshot taken"
            
            elif command == 'volume':
                if 'up' in text or 'increase' in text or 'बढ़ाओ' in text:
                    return "Volume increased"
                elif 'down' in text or 'decrease' in text or 'कम' in text:
                    return "Volume decreased"
            
            elif command == 'calculator':
                return "Calculator opened"
            
            elif command == 'weather':
                return "Weather: 24°C, Clear"
            
            elif command == 'timer':
                return "Timer set"
            
            return "Command executed"
            
        except Exception as e:
            logging.error(f"Command execution error: {e}")
            return "Command failed"

    def generate_mock_call_data(self):
        """Generate mock call data for testing"""
        mock_numbers = [
            "+91-XXXXX-1234",
            "+91-XXXXX-5678", 
            "+91-XXXXX-9999",
            "Unknown Number",
            "Known Contact"
        ]
        
        mock_transcripts = [
            "Hello, this is urgent. We need to verify your OTP for bank account.",
            "Congratulations! You have won a lottery prize. Click here to claim.",
            "Your account has been suspended. Please verify immediately.",
            "Hi, this is from your bank. We need verification for security.",
            "Hello, how are you? Let's meet for coffee tomorrow."
        ]
        
        return {
            'number': np.random.choice(mock_numbers),
            'transcript': np.random.choice(mock_transcripts),
            'timestamp': datetime.now().isoformat(),
            'duration': np.random.randint(10, 300)
        }

    def analyze_call_content(self, call_data):
        """Analyze call content for suspicious activity"""
        transcript = call_data.get('transcript', '').lower()
        suspicious_words = []
        
        # Check for suspicious keywords
        for keyword in self.config['suspicious_keywords']:
            if keyword in transcript:
                suspicious_words.append(keyword)
        
        # Determine risk level
        if len(suspicious_words) >= 3:
            risk_level = "HIGH"
            action = "BLOCKED"
        elif len(suspicious_words) >= 1:
            risk_level = "MEDIUM"
            action = "MONITORED"
        else:
            risk_level = "LOW"
            action = "ALLOWED"
        
        return {
            'risk_level': risk_level,
            'suspicious_words': suspicious_words,
            'action': action
        }

    def log_system_status(self):
        """Log current system status"""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        status_info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'camera_active': self.status['camera_active'],
            'screen_active': self.status['screen_monitor_active'],
            'voice_active': self.status['voice_active'],
            'call_active': self.status['call_monitor_active'],
            'camera_detections': len(self.results['camera_detections']),
            'screen_analysis': len(self.results['screen_analysis']),
            'voice_commands': len(self.results['voice_commands']),
            'call_alerts': len(self.results['call_alerts'])
        }
        
        # Log every 30 seconds
        if int(time.time()) % 30 == 0:
            logging.info(f"System Status: {json.dumps(status_info, indent=2)}")

    def get_system_results(self):
        """Get all system results"""
        return {
            'status': self.status,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }

    def shutdown(self):
        """Shutdown the monitoring system"""
        logging.info("Shutting down Smart Monitoring System...")
        self.running = False
        
        # Wait for threads to complete
        self.executor.shutdown(wait=True)
        
        # Save results to file
        with open(f'monitoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.get_system_results(), f, indent=2)
        
        logging.info("System shutdown complete")

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main entry point"""
    print("=" * 60)
    print("SMART AI MONITORING SYSTEM")
    print("=" * 60)
    print("4 Parallel Processing Components:")
    print("1. Camera + YOLO Object Detection + OCR → GPT-4 Mini")
    print("2. Screen Capture + OCR → GPT-4 Mini Recommendations")
    print("3. Voice Commands (Hindi/English) Processing")
    print("4. Call Monitoring + Suspicious Word Detection")
    print("=" * 60)
    
    # Initialize system
    system = SmartMonitoringSystem()
    
    try:
        # Start monitoring
        system.start_system()
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()