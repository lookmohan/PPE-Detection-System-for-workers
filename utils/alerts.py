import pygame
from gtts import gTTS
import tempfile
import time
import os
import threading
from queue import Queue

# Initialize pygame mixer
pygame.mixer.init()

alert_queue = Queue()

def _play_alert_safe(message):
    """Thread-safe audio playback with cleanup"""
    temp_path = None
    try:
        # Try online TTS first
        tts = gTTS(text=message, lang='en')
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        tts.save(temp_path)
        
        # Wait for file write to complete
        time.sleep(0.2)
        
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"Audio error: {e}")
        # Fallback to offline TTS
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"Offline TTS failed: {e}")
            
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

def _process_alerts():
    """Process alert queue in background"""
    while True:
        message = alert_queue.get()
        _play_alert_safe(message)
        alert_queue.task_done()

# Start alert processor thread
alert_thread = threading.Thread(target=_process_alerts, daemon=True)
alert_thread.start()

def play_alert(message):
    """Public interface for non-blocking alerts"""
    alert_queue.put(message)