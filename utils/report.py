import cv2
import numpy as np
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
import base64
from PIL import Image
import atexit
import shutil

class PPE_Reporter:
    def __init__(self):
        # Create temp directory that auto-cleans
        self.temp_dir = tempfile.mkdtemp()
        atexit.register(self._cleanup)
        
        # Load configurable logo
        self.logo_path = self._get_logo()
        self.template = self._load_template()
    
    def _cleanup(self):
        """Remove temp directory on exit"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def _get_logo(self):
        """Load logo from file or generate default"""
        try:
            if os.path.exists("assets/logo.png"):
                with open("assets/logo.png", "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
        except:
            pass
        
        # Fallback: Generate simple logo
        logo = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(logo, "SAFETYGUARD", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 95, 135), 3)
        _, buffer = cv2.imencode('.png', logo)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _load_template(self):
        return {
            'title': "PPE Compliance Report",
            'subtitle': "Generated by SafetyGuard AI",
            'footer': f"© {datetime.now().year} SafetyGuard - Confidential",
            'required_ppe': ["Helmet", "Safety Vest", "Gloves", "Safety Boots"]
        }
    
    def generate_report(self, output_frame, missing_items, detected_items, report_format='pdf'):
        """Main report generation with proper cleanup"""
        try:
            # Save detection image
            img_path = os.path.join(self.temp_dir, f"detection_{datetime.now().timestamp()}.png")
            cv2.imwrite(img_path, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            
            if not os.path.exists(img_path):
                raise FileNotFoundError("Failed to save detection image")
                
            if report_format.lower() == 'pdf':
                report_path, mime_type = self._generate_pdf(img_path, missing_items, detected_items)
            else:
                report_path, mime_type = self._generate_html(img_path, missing_items, detected_items)
                
            return report_path, mime_type
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            raise
            
    def _generate_pdf(self, image_path, missing_items, detection_data):
        """PDF-specific generation"""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Add Unicode-compatible font
            pdf.add_font('DejaVu', '', 'assets/DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', '', 12)
            
            # Header
            pdf.set_font('DejaVu', '', 16)
            pdf.cell(0, 10, self.template['title'], 0, 1, 'C')
            
            # ... rest of PDF generation code ...
            
            # Save to temp file
            report_path = os.path.join(self.temp_dir, f"report_{datetime.now().timestamp()}.pdf")
            pdf.output(report_path)
            
            return report_path, 'application/pdf'
            
        except Exception as e:
            print(f"PDF generation failed: {e}")
            raise

    def _generate_html(self, image_path, missing_items, detection_data):
        """HTML-specific generation"""
        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
            # ... rest of HTML generation code ...
            
            report_path = os.path.join(self.temp_dir, f"report_{datetime.now().timestamp()}.html")
            with open(report_path, 'w') as f:
                f.write(html_content)
                
            return report_path, 'text/html'
            
        except Exception as e:
            print(f"HTML generation failed: {e}")
            raise