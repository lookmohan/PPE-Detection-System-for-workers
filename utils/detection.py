import cv2

def detect_ppe(model, frame, conf_threshold=0.5):
    """
    Detect PPE equipment with error handling and configurable confidence
    
    Args:
        model: YOLO model instance
        frame: Input image/frame (BGR format)
        conf_threshold: Minimum confidence score (0-1)
        
    Returns:
        tuple: (annotated_frame, missing_items, detected_counts)
    """
    if frame is None or frame.size == 0:
        return frame, [], {}
    
    try:
        # Resize with aspect ratio preservation
        height, width = frame.shape[:2]
        new_height = 640
        new_width = int((new_height / height) * width)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Predict with confidence threshold
        results = model.predict(frame, conf=conf_threshold)
        
        # Handle empty results safely
        if not results or len(results[0]) == 0:
            return frame, ["helmet", "vest", "gloves", "boots"], {}
            
        # Process detections
        detected_classes = []
        if hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            if hasattr(boxes, 'cls'):
                detected_classes = [model.names[int(cls)] for cls in boxes.cls.cpu().numpy()]
        
        # Count detections
        item_counts = {}
        for item in detected_classes:
            item_counts[item] = item_counts.get(item, 0) + 1
        
        # Check required PPE
        required_ppe = ["helmet", "vest", "gloves", "boots"]
        missing = [item for item in required_ppe if item not in item_counts]
        
        # Annotate frame
        output_frame = results[0].plot()
        
        return output_frame, missing, item_counts
        
    except Exception as e:
        print(f"Detection error: {e}")
        return frame, ["helmet", "vest", "gloves", "boots"], {}