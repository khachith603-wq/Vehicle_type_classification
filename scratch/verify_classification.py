import os
import django
import sys

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vehicle_project.settings')
django.setup()

from classifier.utils import VehicleClassifier

def test_prediction():
    image_path = r"C:\Users\vishweshvar\Desktop\vehicle_classify\media\predictions\sports_car_test_1776796543288.png"
    if not os.path.exists(image_path):
        # Try finding it in media if it was uploaded
        print(f"File not found at {image_path}, checking local brain folder...")
        image_path = r"C:\Users\vishweshvar\.gemini\antigravity\brain\a7f8d79f-010b-4d73-8687-a7c301b0af07\sports_car_test_1776796543288.png"

    print(f"Testing classification on: {image_path}")
    try:
        label, confidence = VehicleClassifier.classify_image(image_path)
        print(f"RESULT:")
        print(f"Label: {label}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_prediction()
