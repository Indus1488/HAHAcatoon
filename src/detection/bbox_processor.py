from src.grid_mapping.mapper import GridMapper
from src.config.classes import CLASS_NAMES


class BoundingBoxProcessor:

    def __init__(self):
        self.mapper = GridMapper(
            rows=8,
            cols=8
        )

    def process(self, result, conf_threshold=0.25):
        detections = []

        if result.boxes is None:
            return detections

        if len(result.boxes) == 0:
            return detections

        img_h, img_w = result.orig_shape

        for box in result.boxes:
            conf = float(box.conf[0])

            # === ФИЛЬТРАЦИЯ ПО ПОРОГУ ===
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cls = int(box.cls[0])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            grid = self.mapper.get_grid_location(
                center_x=cx,
                center_y=cy,
                image_width=img_w,
                image_height=img_h
            )

            detections.append({

                "class_name": CLASS_NAMES[cls],

                "confidence": round(conf, 3),

                "bbox": {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                },

                "grid": grid

            })

        return detections