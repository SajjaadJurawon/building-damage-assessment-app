import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from typing import List, Dict, Optional
from ultralytics import YOLO


class YOLOBuildingDamageAssessmentPipeline:
    def __init__(
        self,
        yolo_model_path: str,
        classification_model: torch.nn.Module,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        crop_size: int = 224,
    ):
        # Normalize device
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        # YOLO model (Ultralytics handles device at predict-time; we store preference)
        self.yolo_model = YOLO(yolo_model_path)

        # Classification model
        self.classification_model = classification_model.to(self.device)
        self.classification_model.eval()

        self.confidence_threshold = float(confidence_threshold)
        self.crop_size = int(crop_size)

        # Classification normalization (ImageNet)
        self.cls_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.damage_classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]

        # Prebuild transform (avoid rebuilding per crop)
        self._cls_transform = transforms.Compose(
            [
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                self.cls_normalize,
            ]
        )

    def convert_tif_to_png_for_yolo(self, tif_image: np.ndarray) -> np.ndarray:
        """
        Ensure the image is uint8 RGB and shape compatible for YOLO.
        If tif_image isn't uint8, scale to 0-255.
        """
        img = tif_image

        if img.dtype != np.uint8:
            maxv = float(img.max()) if img.size else 1.0
            if maxv <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = (img / maxv * 255.0).clip(0, 255).astype(np.uint8)

        # Ensure 3 channels if possible (YOLO likes HWC 3)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # If it's RGB already, keep it. If it looks BGR, that's fine too; YOLO handles arrays.
        return img

    def preprocess_crop_for_classification(self, crop: np.ndarray) -> torch.Tensor:
        """
        Convert crop -> PIL -> tensor -> normalized -> add batch dim
        """
        if crop.dtype != np.uint8:
            maxv = float(crop.max()) if crop.size else 1.0
            if maxv <= 1.0:
                crop = (crop * 255.0).clip(0, 255).astype(np.uint8)
            else:
                crop = (crop / maxv * 255.0).clip(0, 255).astype(np.uint8)

        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)

        pil = Image.fromarray(crop)
        return self._cls_transform(pil).unsqueeze(0)

    def detect_buildings_yolo(
        self, image_for_yolo: np.ndarray, reference_image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Stage 1: Detect buildings using YOLO.

        image_for_yolo: the image YOLO sees (often png-like / uint8)
        reference_image: if provided, bbox coords are scaled to match reference_image dims
                         (useful when you detect on a resized/converted image but want bbox on original).
        """
        ref = reference_image if reference_image is not None else image_for_yolo

        # Force YOLO device explicitly
        yolo_device = 0 if self.device.type == "cuda" else "cpu"

        # Use .predict for clarity + consistent args
        results = self.yolo_model.predict(
            source=image_for_yolo,
            conf=self.confidence_threshold,
            device=yolo_device,
            verbose=False,
        )

        # scaling if detection image size differs from reference image size
        scale_x = ref.shape[1] / image_for_yolo.shape[1]
        scale_y = ref.shape[0] / image_for_yolo.shape[0]

        building_detections: List[Dict] = []
        obj_id = 0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                conf = float(box.conf[0].detach().cpu().item())

                # scale into reference coords
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

                # clamp
                x1 = int(max(0, min(x1, ref.shape[1] - 1)))
                y1 = int(max(0, min(y1, ref.shape[0] - 1)))
                x2 = int(max(x1 + 1, min(x2, ref.shape[1])))
                y2 = int(max(y1 + 1, min(y2, ref.shape[0])))

                width = x2 - x1
                height = y2 - y1
                area = width * height
                building_size = max(width, height)
                centroid_x = (x1 + x2) / 2.0
                centroid_y = (y1 + y2) / 2.0

                building_detections.append(
                    {
                        "id": obj_id,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "area": area,
                        "building_size": building_size,
                        "width": width,
                        "height": height,
                        "centroid": (centroid_x, centroid_y),
                    }
                )
                obj_id += 1

        return building_detections

    def extract_building_crops_from_detections(
        self, image: np.ndarray, building_detections: List[Dict]
    ) -> List[Dict]:
        """
        Crop around detection bboxes with padding. Returns updated dicts including 'crop'.
        """
        H, W = image.shape[0], image.shape[1]
        max_crop_size = 512
        padding_ratio = 0.3

        for b in building_detections:
            x1, y1, x2, y2 = b["bbox"]
            building_size = b["building_size"]

            padding = max(16, int(building_size * padding_ratio))

            crop_min_x = max(0, x1 - padding)
            crop_min_y = max(0, y1 - padding)
            crop_max_x = min(W, x2 + padding)
            crop_max_y = min(H, y2 + padding)

            crop_w = crop_max_x - crop_min_x
            crop_h = crop_max_y - crop_min_y
            crop_size = max(crop_w, crop_h)

            if crop_size > max_crop_size:
                excess = crop_size - max_crop_size
                reduced_padding = max(8, padding - (excess // 2))

                crop_min_x = max(0, x1 - reduced_padding)
                crop_min_y = max(0, y1 - reduced_padding)
                crop_max_x = min(W, x2 + reduced_padding)
                crop_max_y = min(H, y2 + reduced_padding)

                crop_w = crop_max_x - crop_min_x
                crop_h = crop_max_y - crop_min_y

                if max(crop_w, crop_h) > max_crop_size:
                    cx, cy = b["centroid"]
                    half = max_crop_size // 2

                    crop_min_x = max(0, int(cx - half))
                    crop_min_y = max(0, int(cy - half))
                    crop_max_x = min(W, crop_min_x + max_crop_size)
                    crop_max_y = min(H, crop_min_y + max_crop_size)

                    if crop_max_x - crop_min_x < max_crop_size:
                        crop_min_x = max(0, crop_max_x - max_crop_size)
                    if crop_max_y - crop_min_y < max_crop_size:
                        crop_min_y = max(0, crop_max_y - max_crop_size)

                crop_w = crop_max_x - crop_min_x
                crop_h = crop_max_y - crop_min_y
                crop_size = max(crop_w, crop_h)

            crop = image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
            ratio = building_size / crop_size if crop_size > 0 else 0.0

            b.update(
                {
                    "crop": crop,
                    "crop_bbox": (crop_min_x, crop_min_y, crop_max_x, crop_max_y),
                    "crop_size": crop_size,
                    "building_ratio": ratio,
                    "padding_used": padding,
                }
            )

        return building_detections

    def classify_damage(self, building_detections: List[Dict]) -> List[Dict]:
        """
        Stage 2: classify each crop with EfficientNet.
        """
        if not building_detections:
            return building_detections

        with torch.no_grad():
            for b in building_detections:
                crop = b.get("crop", None)
                if crop is None or crop.size == 0:
                    b["damage_class"] = "no-damage"
                    b["damage_confidence"] = 0.0
                    b["damage_probabilities"] = np.zeros((4,), dtype=np.float32)
                    continue

                x = self.preprocess_crop_for_classification(crop).to(self.device)
                logits = self.classification_model(x)

                probs = F.softmax(logits, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0, pred].item())

                b["damage_class"] = self.damage_classes[pred]
                b["damage_confidence"] = conf
                b["damage_probabilities"] = probs[0].detach().cpu().numpy()

        return building_detections

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Full pipeline: detect + crop + classify.
        Accepts either tif or rgb uint8; will convert for YOLO when needed.
        """
        tif_image = image
        png_image = self.convert_tif_to_png_for_yolo(tif_image)

        building_detections = self.detect_buildings_yolo(png_image, reference_image=tif_image)
        building_detections = self.extract_building_crops_from_detections(tif_image, building_detections)
        building_detections = self.classify_damage(building_detections)

        results = {
            "original_tif_image": tif_image,
            "png_detection_image": png_image,
            "buildings": building_detections,
            "num_buildings": len(building_detections),
            "damage_summary": self._create_damage_summary(building_detections),
            "coordinate_scaling_applied": (png_image.shape[:2] != tif_image.shape[:2]),
        }
        return results

    def _create_damage_summary(self, building_detections: List[Dict]) -> Dict:
        if not building_detections:
            return {"total_buildings": 0, "total_area": 0, "damage_counts": {c: 0 for c in self.damage_classes},
                    "damage_area": {c: 0 for c in self.damage_classes}, "damage_percentages": {c: 0.0 for c in self.damage_classes}}

        damage_counts = {c: 0 for c in self.damage_classes}
        damage_area = {c: 0 for c in self.damage_classes}
        total_area = 0

        for b in building_detections:
            cls = b.get("damage_class", "no-damage")
            area = int(b.get("area", 0))
            damage_counts[cls] = damage_counts.get(cls, 0) + 1
            damage_area[cls] = damage_area.get(cls, 0) + area
            total_area += area

        total = len(building_detections)
        damage_percentages = {c: (damage_counts.get(c, 0) / total * 100.0) for c in self.damage_classes}

        return {
            "total_buildings": total,
            "total_area": total_area,
            "damage_counts": damage_counts,
            "damage_area": damage_area,
            "damage_percentages": damage_percentages,
        }

    def results_to_json(self, results: Dict) -> Dict:
        buildings_json = []
        for b in results["buildings"]:
            probs = b.get("damage_probabilities", np.zeros((4,), dtype=np.float32))

            buildings_json.append(
                {
                    "id": int(b["id"]),
                    "bbox": {
                        "x1": int(b["bbox"][0]),
                        "y1": int(b["bbox"][1]),
                        "x2": int(b["bbox"][2]),
                        "y2": int(b["bbox"][3]),
                    },
                    "detection_confidence": float(b.get("confidence", 0.0)),
                    "area": int(b.get("area", 0)),
                    "dimensions": {"width": int(b.get("width", 0)), "height": int(b.get("height", 0))},
                    "centroid": {"x": float(b["centroid"][0]), "y": float(b["centroid"][1])},
                    "damage_assessment": {
                        "class": b.get("damage_class", "no-damage"),
                        "confidence": float(b.get("damage_confidence", 0.0)),
                        "probabilities": {
                            "no-damage": float(probs[0]) if len(probs) > 0 else 0.0,
                            "minor-damage": float(probs[1]) if len(probs) > 1 else 0.0,
                            "major-damage": float(probs[2]) if len(probs) > 2 else 0.0,
                            "destroyed": float(probs[3]) if len(probs) > 3 else 0.0,
                        },
                    },
                }
            )

        summary = results["damage_summary"]

        return {
            "success": True,
            "metadata": {
                "total_buildings": int(summary.get("total_buildings", 0)),
                "image_dimensions": {
                    "height": int(results["original_tif_image"].shape[0]),
                    "width": int(results["original_tif_image"].shape[1]),
                },
            },
            "damage_summary": {
                "counts": summary.get("damage_counts", {}),
                "percentages": {k: round(v, 2) for k, v in summary.get("damage_percentages", {}).items()},
                "total_area_pixels": int(summary.get("total_area", 0)),
            },
            "buildings": buildings_json,
        }


def create_yolo_pipeline_from_saved_models(
    yolo_model_path: str,
    classification_model_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.1,
) -> YOLOBuildingDamageAssessmentPipeline:
    """
    Build EfficientNet-B3 and load weights, then create pipeline.
    """
    from torchvision.models import efficientnet_b3

    classification_model = efficientnet_b3(weights=None)
    in_feats = classification_model.classifier[1].in_features
    classification_model.classifier[1] = nn.Linear(in_feats, 4)

    torch_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(classification_model_path, map_location=torch_device)

    if isinstance(ckpt, dict) and any(k in ckpt for k in ["state_dict", "model_state_dict"]):
        ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict"))

    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    classification_model.load_state_dict(ckpt, strict=True)

    return YOLOBuildingDamageAssessmentPipeline(
        yolo_model_path=yolo_model_path,
        classification_model=classification_model,
        device=device,
        confidence_threshold=confidence_threshold,
        crop_size=224,
    )

