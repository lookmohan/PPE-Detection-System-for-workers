import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING'] = 'false'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
from utils.detection import detect_ppe
from utils.alerts import play_alert
from utils.report import PPE_Reporter



# Initialize app
def initialize_app():
    try:
        from ultralytics import YOLO
        import torch
        model = torch.load("models/best.pt", weights_only=False)
        if torch.cuda.is_available():
            model.to('cuda')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load custom CSS
def cssload():
    with open("style.css") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# ============================================
# 🚀 Main App UI
# ============================================


def main():
    # Show initial loading animation
    cssload()
    # Initialize app
    model = initialize_app()
    if model is None:
        st.error("Failed to initialize PPE detection model")
        print("🔴 Model loading failed")
        return
    print("🟢 Model loaded successfully")
    
    # Main header with logo
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--safety-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        </svg>
        <h1 style="color: var(--safety-primary); margin: 0;">SafetyGuard <span style="font-size: 0.8em; color: var(--safety-secondary);">AI</span></h1>
    </div>
    """, unsafe_allow_html=True)

    # PPE Compliance Monitoring Card
    st.markdown("""
    <div class="card">
        <h3 style="color: var(--safety-primary); margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            PPE Compliance Monitoring
        </h3>
        <p style="color: #495057;">Advanced detection of essential safety equipment for industrial workers:</p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 15px;">
            <div class="ppe-tag" style="background: #E3F2FD; color: #0D47A1;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                    <path d="M2 18v3c0 .6.4 1 1 1h4v-3h3v3h4.5c.3 0 .5-.2.5-.5v-1.5c0-.3-.2-.5-.5-.5H13v-3h-3v3H7v-3H3c-.6 0-1 .4-1 1z"></path>
                    <path d="M10 10V5c0-1.1.9-2 2-2h1c1.1 0 2 .9 2 2v5"></path>
                    <path d="M5 12c-1.7 0-3-1.3-3-3v-1a2 2 0 0 1 2-2h3v5H5z"></path>
                    <path d="M19 12c1.7 0 3-1.3 3-3v-1a2 2 0 0 0-2-2h-3v5h2z"></path>
                </svg>
                Hard Hat
            </div>
            <div class="ppe-tag" style="background: #E8F5E9; color: #2E7D32;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                    <path d="M4 12h8m4 0h4"></path>
                    <path d="M18 16v4a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2v-4"></path>
                    <path d="M18 8V4a2 2 0 0 0-2-2h-4"></path>
                    <path d="M6 8V4a2 2 0 0 1 2-2h4"></path>
                    <path d="M11 8h2"></path>
                    <path d="M11 12h2"></path>
                    <path d="M11 16h2"></path>
                </svg>
                Safety Vest
            </div>
            <div class="ppe-tag" style="background: #FFEBEE; color: #C62828;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                    <path d="M20 17a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3.9a2 2 0 0 1-1.69-.9l-.81-1.2a2 2 0 0 0-1.67-.9H9.6a2 2 0 0 0-1.68.9l-.8 1.2A2 2 0 0 1 6 7H2"></path>
                    <path d="M3 8v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8"></path>
                    <path d="M7 13h10"></path>
                    <path d="M12 10v4"></path>
                </svg>
                Gloves
            </div>
            <div class="ppe-tag" style="background: #F3E5F5; color: #7B1FA2;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                    <path d="M4 12h16"></path>
                    <path d="M8 12v8a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-8"></path>
                    <path d="M10 12V5a2 2 0 0 1 2-2h0a2 2 0 0 1 2 2v7"></path>
                    <path d="M18 12V5a2 2 0 0 0-2-2h0a2 2 0 0 0-2 2v7"></path>
                </svg>
                Safety Boots
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with settings
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: black; display: flex; align-items: center; justify-content: center; gap: 8px;">
                Settings
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        enable_audio = st.checkbox("🔊 Enable Voice Alerts", value=True)

        st.markdown("""
        <div style="color: black;">
            <h4 style="display: flex; align-items: center; gap: 8px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                    <line x1="12" y1="17" x2="12" y2="17"></line>
                </svg>
                About SafetyGuard
            </h4>
            <p style="font-size: 0.9rem;">AI-powered workplace safety monitoring system that detects PPE compliance in real-time.</p>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;">
                <p style="font-size: 0.9rem; margin-bottom: 4px;"><strong>Version:</strong> 3.0.0</p>
                <p style="font-size: 0.9rem; margin-bottom: 4px;"><strong>Model:</strong> YOLOv8 (Custom)</p>
                <p style="font-size: 0.9rem;"><strong>Accuracy:</strong> 94.2% mAP</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    tab1, tab2 = st.tabs(["📷 Image Inspection", "🎥 Live Inspection"])
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 style="color: var(--safety-primary); display: flex; align-items: center; gap: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                Upload Worker Photo
            </h3>
            <p style="color: #495057;">Analyze PPE compliance from uploaded images of workers</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], 
                                       help="Upload a clear photo of workers to analyze PPE compliance")
        
        if uploaded_file is not None:
            try:
                # Custom loading animation
                with st.spinner(""):
                    loading_placeholder = st.empty()
                    loading_placeholder.markdown("""
                    <div class="loading-container">
                        <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json" 
                                     background="transparent" speed="1" style="width: 120px; height: 120px;" loop autoplay>
                        </lottie-player>
                        <p style="color: var(--safety-primary); font-weight: 500; margin-top: 15px; text-align: center;">
                            Analyzing PPE Compliance<br>
                            <span style="font-size: 0.9rem; color: #6c757d;">Processing image with AI model...</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Process image
                    image = Image.open(uploaded_file).convert("RGB")
                    frame = np.array(image)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    output_frame, missing, detected_items = detect_ppe(model, frame)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    
                    # Clear loading animation
                    loading_placeholder.empty()
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Original Image")
                        st.image(image, use_container_width=True, caption="Uploaded Image")
                    with col2:
                        st.markdown("#### PPE Detection")
                        st.image(output_frame, use_container_width=True, caption="AI Analysis Results")
                    
                    # Results card
                    if missing:
                        st.markdown(f"""
                        <div class="card pulse" style="border-left: 4px solid var(--safety-accent);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <lottie-player src="https://assets9.lottiefiles.com/packages/lf20_bujdzzbk.json" 
                                           background="transparent" speed="1" style="width: 50px; height: 50px;" loop autoplay>
                                </lottie-player>
                                <div>
                                    <h3 style="color: var(--safety-accent); margin: 0;">Safety Violation Detected</h3>
                                    <p style="color: #495057; font-size: 0.9rem; margin: 0;">{len(missing)} PPE items missing</p>
                                </div>
                            </div>
                            <p style="color: #495057;">Missing safety equipment:</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                                {' '.join([f'<div class="ppe-tag" style="background: #FFEBEE; color: var(--safety-accent);">{item.title()}</div>' for item in missing])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if enable_audio:
                            play_alert(f"Safety alert! Missing equipment: {', '.join(missing)}")
                    else:
                        st.markdown("""
                        <div class="card" style="border-left: 4px solid var(--safety-success);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_sk5h1kfn.json" 
                                           background="transparent" speed="1" style="width: 50px; height: 50px;" loop autoplay>
                                </lottie-player>
                                <div>
                                    <h3 style="color: var(--safety-success); margin: 0;">Full PPE Compliance</h3>
                                    <p style="color: #495057; font-size: 0.9rem; margin: 0;">All required equipment detected</p>
                                </div>
                            </div>
                            <p style="color: #495057;">Worker is properly equipped with all required safety gear.</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                                <div class="ppe-tag" style="background: #E8F5E9; color: var(--safety-success);">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px;">
                                        <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                    Compliant
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if enable_audio:
                            play_alert("All safety equipment detected. Good compliance.")
                    
                    # Report generation
                    st.markdown("---")
                    st.markdown("""
                    <div class="card">
                        <h3 style="color: var(--safety-primary); display: flex; align-items: center; gap: 10px;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10 9 9 9 8 9"></polyline>
                            </svg>
                            Generate Safety Report
                        </h3>
                        <p style="color: #495057;">Document this inspection for your records</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    report_format = st.radio("Select report format:", ["PDF", "HTML"], horizontal=True, 
                                            help="Choose between PDF (printable) or HTML (interactive) report formats")
                    
                    if st.button("📄 Generate Report", key="report_btn"):
                        with st.spinner(f"Generating {report_format} report..."):
                            reporter = PPE_Reporter()
                            report_path, mime_type = reporter.generate_report(
                                output_frame, 
                                missing, 
                                detected_items,
                                report_format.lower()
                            )
                            
                            with open(report_path, "rb") as f:
                                btn = st.download_button(
                                    label=f"⬇️ Download {report_format} Report",
                                    data=f,
                                    file_name=f"PPE_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}",
                                    mime=mime_type
                                )
                            
                            os.unlink(report_path)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 style="color: var(--safety-primary); display: flex; align-items: center; gap: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="23 7 16 12 23 17 23 7"></polygon>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg>
                Live Camera Inspection
            </h3>
            <p style="color: #495057;">Real-time PPE monitoring using your webcam</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Note:** Live inspection requires camera access. 
        - Ensure proper lighting for best results
        - Position worker clearly in frame
        - System will alert for missing PPE
        """)
        
        if st.button("▶️ Start Live Inspection", key="live_start"):
            cam = None
            frame_placeholder = st.empty()
            stop_button = st.button("⏹️ Stop Inspection")
            last_alert_time = 0
            alert_cooldown = 5  # seconds
            frame_counter = 0
            processing_frame = False
            
            try:
                # Initialize camera with error handling
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    st.error("Failed to access camera")
                    return
                    
                while not stop_button:
                    # Skip frames for better performance
                    frame_counter += 1
                    if frame_counter % 3 != 0:  # Process every 3rd frame
                        continue
                        
                    # Read frame safely
                    ret, frame = cam.read()
                    if not ret:
                        st.warning("Camera disconnected")
                        break
                        
                    try:
                        processing_frame = True
                        
                        # Process frame with error handling
                        output_frame, missing, _ = detect_ppe(model, frame)
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display results
                        frame_placeholder.image(output_frame, channels="RGB", 
                                            use_container_width=True,
                                            caption="Live PPE Detection - Worker View")
                        
                        # Handle alerts
                        current_time = time.time()
                        if missing:
                            violation_html = f"""
                            <div style="background: #FFF3E0; padding: 15px; border-radius: 8px; 
                                        border-left: 4px solid #FFA000; margin: 10px 0;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" 
                                        viewBox="0 0 24 24" fill="none" stroke="#FFA000" 
                                        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <circle cx="12" cy="12" r="10"></circle>
                                        <line x1="12" y1="8" x2="12" y2="12"></line>
                                        <line x1="12" y1="16" x2="12" y2="16"></line>
                                    </svg>
                                    <h4 style="color: #FFA000; margin: 0;">Safety Violation Detected</h4>
                                </div>
                                <p style="color: #5D4037; margin: 8px 0 0 0;">
                                    Missing equipment: {', '.join(missing)}
                                </p>
                            </div>
                            """
                            st.markdown(violation_html, unsafe_allow_html=True)
                            
                            if enable_audio and (current_time - last_alert_time) > alert_cooldown:
                                play_alert_async(f"Warning! Missing safety equipment: {', '.join(missing)}")
                                last_alert_time = current_time
                        else:
                            success_html = """
                            <div style="background: #E8F5E9; padding: 15px; border-radius: 8px; 
                                        border-left: 4px solid #4CAF50; margin: 10px 0;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" 
                                        viewBox="0 0 24 24" fill="none" stroke="#4CAF50" 
                                        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                                    </svg>
                                    <h4 style="color: #4CAF50; margin: 0;">All PPE Detected</h4>
                                </div>
                                <p style="color: #2E7D32; margin: 8px 0 0 0;">
                                    Worker is properly equipped
                                </p>
                            </div>
                            """
                            st.markdown(success_html, unsafe_allow_html=True)
                            
                            if enable_audio and (current_time - last_alert_time) > alert_cooldown:
                                play_alert_async("All safety equipment detected")
                                last_alert_time = current_time
                                
                        processing_frame = False
                        
                    except Exception as e:
                        st.error(f"Frame processing error: {str(e)}")
                        processing_frame = False
                        continue
                        
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                
            finally:
                # Ensure clean shutdown
                if cam is not None:
                    # Wait for current frame to finish processing
                    while processing_frame:
                        time.sleep(0.1)
                        
                    cam.release()
                    st.info("Live inspection stopped. Camera resources released.")
                    
                # Clear the frame placeholder
                frame_placeholder.empty()

if __name__ == "__main__":
    main()