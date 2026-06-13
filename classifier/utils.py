import os
import tempfile
import shutil
from PIL import Image
from transformers import pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------------------------------------------------------------------
# ImageNet labels that map to broad vehicle categories.
# The general ViT (vit-base-patch16-224-in21k fine-tuned on ImageNet-1k)
# uses these exact label strings.
# ---------------------------------------------------------------------------
VEHICLE_CATEGORY_MAP = {
    # Two-wheelers
    "motorcycle":           "Motorcycle",
    "motor scooter":        "Scooter",
    "moped":                "Moped",
    "bicycle":              "Bicycle",
    "mountain bike":        "Mountain Bike",
    "tricycle":             "Tricycle",

    # Cars / light vehicles  → hand off to the Stanford-Cars specialist
    "sports car":           "__CAR__",
    "race car":             "__CAR__",
    "car":                  "__CAR__",
    "sedan":                "__CAR__",
    "convertible":          "__CAR__",
    "limousine":            "__CAR__",
    "taxicab":              "Taxi / Cab",
    "cab":                  "Taxi / Cab",
    "minivan":              "Minivan",
    "station wagon":        "Station Wagon",
    "jeep":                 "__CAR__",
    "beach wagon":          "SUV / Wagon",
    "ambulance":            "Ambulance",

    # Heavy / commercial
    "bus":                  "Bus",
    "minibus":              "Minibus",
    "trolleybus":           "Trolleybus",
    "school bus":           "School Bus",
    "truck":                "Truck",
    "pickup":               "Pickup Truck",
    "fire engine":          "Fire Truck",
    "fire truck":           "Fire Truck",
    "garbage truck":        "Garbage Truck",
    "moving van":           "Moving Van",
    "trailer truck":        "Semi-Truck / Trailer",
    "semi":                 "Semi-Truck / Trailer",
    "tractor":              "Tractor",

    # Rail / other
    "streetcar":            "Tram / Streetcar",
    "electric locomotive":  "Train / Locomotive",
    "steam locomotive":     "Train / Locomotive",
    "freight car":          "Train Freight Car",
    "passenger car":        "Train Passenger Car",

    # Water / air (bonus)
    "speedboat":            "Speedboat",
    "catamaran":            "Catamaran",
    "aircraft carrier":     "Aircraft Carrier",
    "airliner":             "Airliner",
    "helicopter":           "Helicopter",
}

# Any ImageNet label that *contains* one of these substrings is also a vehicle
VEHICLE_KEYWORD_HINTS = [
    "car", "truck", "bus", "bike", "cycle", "motorcycle", "scooter",
    "van", "cab", "taxi", "jeep", "wagon", "ambulance", "tractor",
    "train", "locomotive", "tram", "boat", "vehicle", "auto",
]


def _is_vehicle_label(label: str) -> bool:
    low = label.lower()
    if low in VEHICLE_CATEGORY_MAP:
        return True
    return any(kw in low for kw in VEHICLE_KEYWORD_HINTS)


def _map_label(label: str) -> str:
    """Return the friendly category string, or '__CAR__' for the specialist path."""
    low = label.lower()
    # Exact map first
    if low in VEHICLE_CATEGORY_MAP:
        return VEHICLE_CATEGORY_MAP[low]
    # Substring fallback
    for kw, category in VEHICLE_CATEGORY_MAP.items():
        if kw in low:
            return category
    return label.replace("_", " ").title()


class VehicleClassifier:
    _general_model = None   # broad ImageNet classifier
    _car_model = None        # Stanford-Cars specialist

    # ------------------------------------------------------------------
    # Model loaders (lazy singletons)
    # ------------------------------------------------------------------
    @classmethod
    def get_general_model(cls):
        if cls._general_model is None:
            # google/vit-base-patch16-224 – trained on ImageNet-21k + fine-tuned
            # on ImageNet-1k; knows 1 000 classes incl. all common vehicle types
            cls._general_model = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
            )
        return cls._general_model

    @classmethod
    def get_car_model(cls):
        if cls._car_model is None:
            # Specialised ViT fine-tuned on Stanford Cars (190 + make/model classes)
            cls._car_model = pipeline(
                "image-classification",
                model="therealcyberlord/stanford-car-vit-patch16",
            )
        return cls._car_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def classify_image(cls, image_source):
        """
        Two-stage classification:
          Stage 1 – general model identifies the vehicle type.
          Stage 2 – if it looks like a car, the Stanford-Cars specialist
                    gives the precise make / model / year.

        Returns (label, confidence, vehicle_type, detection_stage)
        where detection_stage is 'general' or 'specialist'.
        """
        temp_path = None
        try:
            # ── Write image to a temp file ──────────────────────────────
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)

            if hasattr(image_source, 'read'):
                image_source.seek(0)
                with open(temp_path, 'wb') as f:
                    f.write(image_source.read())
            elif isinstance(image_source, str) and os.path.exists(image_source):
                shutil.copy2(image_source, temp_path)
            else:
                raise ValueError("Invalid image source provided")

            # Validate image
            with Image.open(temp_path) as img:
                img.verify()

            # ── Stage 1 : general classification ───────────────────────
            general_results = cls.get_general_model()(temp_path, top_k=5)

            # Find the highest-scoring result that is a vehicle
            vehicle_hit = None
            for result in general_results:
                if _is_vehicle_label(result['label']):
                    vehicle_hit = result
                    break

            if vehicle_hit is None:
                # Top result isn't a vehicle – still return it but flag it
                top = general_results[0]
                label = top['label'].replace('_', ' ').title()
                confidence = float(top['score']) * 100
                return label, confidence, "Unknown / Non-Vehicle", "general"

            mapped = _map_label(vehicle_hit['label'])
            general_confidence = float(vehicle_hit['score']) * 100

            # ── Stage 2 : if it's a car, use the specialist ─────────────
            if mapped == "__CAR__":
                car_results = cls.get_car_model()(temp_path)
                top_car = car_results[0]
                label      = top_car['label'].replace('_', ' ').title()
                confidence = float(top_car['score']) * 100
                vehicle_type = "Car"
                stage = "specialist"
            else:
                label        = mapped
                confidence   = general_confidence
                vehicle_type = mapped
                stage        = "general"

            return label, confidence, vehicle_type, stage

        except Exception as e:
            raise RuntimeError(f"AI Engine Error: {str(e)}")

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
