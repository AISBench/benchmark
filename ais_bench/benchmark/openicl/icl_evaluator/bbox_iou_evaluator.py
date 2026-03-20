import json
import math

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS


def _compute_iou(box1: list, box2: list) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    inter = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _round_by_factor(value: float, factor: int) -> int:
    return max(factor, round(value / factor) * factor)


def _floor_by_factor(value: float, factor: int) -> int:
    return max(factor, math.floor(value / factor) * factor)


def _ceil_by_factor(value: float, factor: int) -> int:
    return max(factor, math.ceil(value / factor) * factor)


def _smart_resize(
    height: float,
    width: float,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 16 * 16 * 4 * 16384,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {height} / {width}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


@ICL_EVALUATORS.register_module()
class BBoxIoUEvaluator(BaseEvaluator):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        coord_scale: float = 1000.0,
        reference_bbox_key: str = "bbox",
        image_width_key: str = "image_width",
        image_height_key: str = "image_height",
        metric_prefix: str = "Accuracy",
        clip_to_image: bool = True,
        smart_resize_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.coord_scale = coord_scale
        self.reference_bbox_key = reference_bbox_key
        self.image_width_key = image_width_key
        self.image_height_key = image_height_key
        self.metric_prefix = metric_prefix
        self.clip_to_image = clip_to_image
        self.smart_resize_cfg = dict(smart_resize_cfg) if smart_resize_cfg else None

    def _scale_prediction(
        self, pred_box: list, image_width: float, image_height: float
    ) -> list:
        if len(pred_box) != 4:
            raise ValueError("Predicted bbox must contain four coordinates")

        scaled_box = [
            float(pred_box[0]) / self.coord_scale * float(image_width),
            float(pred_box[1]) / self.coord_scale * float(image_height),
            float(pred_box[2]) / self.coord_scale * float(image_width),
            float(pred_box[3]) / self.coord_scale * float(image_height),
        ]

        if self.smart_resize_cfg is not None:
            resize_height, resize_width = _smart_resize(
                height=image_height,
                width=image_width,
                factor=self.smart_resize_cfg.get("factor", 32),
                min_pixels=self.smart_resize_cfg.get("min_pixels", 32 * 32 * 16),
                max_pixels=self.smart_resize_cfg.get("max_pixels", 32 * 32 * 4 * 16384),
            )
            height_scale = resize_height / image_height
            width_scale = resize_width / image_width
            scaled_box = [
                scaled_box[0] * width_scale,
                scaled_box[1] * height_scale,
                scaled_box[2] * width_scale,
                scaled_box[3] * height_scale,
            ]

        if self.clip_to_image:
            scaled_box = [
                min(max(scaled_box[0], 0.0), float(image_width)),
                min(max(scaled_box[1], 0.0), float(image_height)),
                min(max(scaled_box[2], 0.0), float(image_width)),
                min(max(scaled_box[3], 0.0), float(image_height)),
            ]

        if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
            raise ValueError("Predicted bbox is reversed or empty after scaling")
        return scaled_box

    def score(self, predictions, references):  # pyright: ignore[reportIncompatibleMethodOverride]
        if len(predictions) != len(references):
            return {
                "error": "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }

        details = []
        scores = []
        for pred, ref in zip(predictions, references):
            detail = {
                "pred": pred,
                "answer": ref,
                "correct": False,
                "coord_mode": "smart_resize_pixels"
                if self.smart_resize_cfg
                else f"0-{int(self.coord_scale)}",
            }

            try:
                refer = json.loads(ref) if isinstance(ref, str) else ref
                image_width = float(refer[self.image_width_key])
                image_height = float(refer[self.image_height_key])
                pred_box_pixel = self._scale_prediction(pred, image_width, image_height)
                gt_box = [float(value) for value in refer[self.reference_bbox_key]]

                iou = _compute_iou(pred_box_pixel, gt_box)
                correct = iou >= self.iou_threshold
                detail["correct"] = correct
                detail["iou"] = iou
                detail["pred_bbox_pixel"] = pred_box_pixel
                scores.append(1 if correct else 0)
            except (
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
                IndexError,
            ) as error:
                detail["iou"] = 0.0
                detail["pred_bbox_pixel"] = None
                detail["invalid"] = True
                detail["error"] = str(error)
                scores.append(0)

            details.append(detail)

        return {
            f"{self.metric_prefix}@{self.iou_threshold}": 100
            * sum(scores)
            / len(scores)
            if scores
            else 0.0,
            "details": details,
        }
