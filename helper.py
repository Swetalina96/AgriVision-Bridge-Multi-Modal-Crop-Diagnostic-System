from ultralytics import YOLO

def load_model(model_path):
    """
    Load YOLO model from weights path
    """
    return YOLO(model_path)
