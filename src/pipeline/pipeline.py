import os

from src.preprocessing.preprocessor import Preprocessor
from src.detection.predictor import Predictor
from src.detection.bbox_processor import BoundingBoxProcessor
from src.visualization.draw import Visualizer

from src.analysis.health import PCBHealthAnalyzer
from src.analysis.recommendation import RecommendationEngine
from src.analysis.report import ReportGenerator


class PCBInspectionPipeline:
    """
    Complete AI PCB Inspection Pipeline

    Pipeline Flow:
    Image
        ↓
    Preprocessing
        ↓
    YOLO Prediction
        ↓
    Bounding Box Processing
        ↓
    Grid Mapping
        ↓
    Visualization
        ↓
    Health Analysis
        ↓
    Recommendation Generation
        ↓
    Report Generation
    """

    def __init__(self, model_path):

        print("Initializing PCB Inspection Pipeline...")

        self.preprocessor = Preprocessor()

        self.predictor = Predictor(model_path)

        self.processor = BoundingBoxProcessor()

        self.visualizer = Visualizer()

        self.health = PCBHealthAnalyzer()

        self.recommendation = RecommendationEngine()

        self.report = ReportGenerator()

        print("Pipeline Ready!")

    def run(
        self,
        image_path,
        confidence=0.25
    ):

        # ----------------------------------------
        # Load Image
        # ----------------------------------------

        image = self.preprocessor.load_image(image_path)

        # ----------------------------------------
        # YOLO Prediction
        # ----------------------------------------

        results = self.predictor.predict(
            image_path=image_path,
            save=False,
            conf=confidence
        )

        result = results[0]

        # ----------------------------------------
        # Process Bounding Boxes
        # ----------------------------------------

        detections = self.processor.process(result, conf_threshold=confidence)

        # ----------------------------------------
        # Draw Grid
        # ----------------------------------------

        image = self.visualizer.draw_grid(image)

        # ----------------------------------------
        # Draw Bounding Boxes
        # ----------------------------------------

        image = self.visualizer.draw_detections(
            image,
            detections
        )

        # ----------------------------------------
        # PCB Health
        # ----------------------------------------

        health = self.health.analyze(
            detections
        )

        # ----------------------------------------
        # Recommendations
        # ----------------------------------------

        recommendations = self.recommendation.generate(
            detections
        )

        # ----------------------------------------
        # Report
        # ----------------------------------------

        report = self.report.generate(
            image_name=os.path.basename(image_path),
            detections=detections,
            health=health,
            recommendations=recommendations
        )

        # ----------------------------------------
        # Return Everything
        # ----------------------------------------

        return {

            "image": image,

            "detections": detections,

            "health": health,

            "recommendations": recommendations,

            "report": report

        }