import os
from src.detection.predictor import Predictor

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/best.pt"

TEST_FOLDER = "datasets/test/images"

SAVE_RESULTS = True

# ==========================================
# Load Model
# ==========================================

predictor = Predictor(MODEL_PATH)

# ==========================================
# Get All Test Images
# ==========================================

image_extensions = (".jpg", ".jpeg", ".png")

image_files = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith(image_extensions)
])

print("\n========================================")
print("PCB DEFECT DETECTION TEST")
print("========================================")
print(f"Total Images Found : {len(image_files)}")

# ------------------------------------------
# Test only first 30 images
# ------------------------------------------

image_files = image_files[:30]

print(f"Images Selected : {len(image_files)}")

# ==========================================
# Statistics
# ==========================================

total_detections = 0
images_without_detection = 0

class_counter = {
    "mouse_bite": 0,
    "spur": 0,
    "missing_hole": 0,
    "short": 0,
    "open_circuit": 0,
    "spurious_copper": 0
}

# ==========================================
# Predict
# ==========================================

for index, image_name in enumerate(image_files, start=1):

    image_path = os.path.join(TEST_FOLDER, image_name)

    print("\n========================================")
    print(f"Image {index}/{len(image_files)}")
    print(image_name)
    print("========================================")

    results = predictor.predict(
        image_path,
        save=SAVE_RESULTS
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:

        print(f"✅ {image_name} — РАБОЧАЯ (дефектов нет)")

        images_without_detection += 1

        continue

    else:

        print(f"❌ {image_name} — НЕРАБОЧАЯ ({len(result.boxes)} дефектов)")

    print(f"✅ Detections : {len(result.boxes)}")

    total_detections += len(result.boxes)

    for i in range(len(result.boxes)):

        cls_id = int(result.boxes.cls[i])

        conf = float(result.boxes.conf[i])

        coords = result.boxes.xyxy[i].tolist()

        class_name = result.names[cls_id]

        class_counter[class_name] += 1

        print("--------------------------------")

        print("Class       :", class_name)

        print("Confidence  :", round(conf, 3))

        print("Coordinates :", [round(x, 2) for x in coords])

# ==========================================
# Summary
# ==========================================

print("\n")
print("========================================")
print("FINAL REPORT")
print("========================================")

print("Images Tested :", len(image_files))

print("Total Detections :", total_detections)

print("Images Without Detection :", images_without_detection)

print("\nDetection Count")

for key, value in class_counter.items():
    print(f"{key:20} : {value}")

print("\n✅ Testing Completed Successfully!")