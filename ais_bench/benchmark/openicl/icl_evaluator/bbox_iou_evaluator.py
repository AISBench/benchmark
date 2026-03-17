import json

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


@ICL_EVALUATORS.register_module()
class BBoxIoUEvaluator(BaseEvaluator):

    def __init__(self,
                 iou_threshold: float = 0.5,
                 coord_scale: float = 1000.0,
                 reference_bbox_key: str = 'bbox',
                 image_width_key: str = 'image_width',
                 image_height_key: str = 'image_height',
                 metric_prefix: str = 'Accuracy',
                 clip_to_image: bool = True) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.coord_scale = coord_scale
        self.reference_bbox_key = reference_bbox_key
        self.image_width_key = image_width_key
        self.image_height_key = image_height_key
        self.metric_prefix = metric_prefix
        self.clip_to_image = clip_to_image

    def _scale_prediction(self, pred_box: list, image_width: float, image_height: float) -> list:
        if len(pred_box) != 4:
            raise ValueError('Predicted bbox must contain four coordinates')

        scaled_box = [
            float(pred_box[0]) / self.coord_scale * float(image_width),
            float(pred_box[1]) / self.coord_scale * float(image_height),
            float(pred_box[2]) / self.coord_scale * float(image_width),
            float(pred_box[3]) / self.coord_scale * float(image_height),
        ]

        if self.clip_to_image:
            scaled_box = [
                min(max(scaled_box[0], 0.0), float(image_width)),
                min(max(scaled_box[1], 0.0), float(image_height)),
                min(max(scaled_box[2], 0.0), float(image_width)),
                min(max(scaled_box[3], 0.0), float(image_height)),
            ]

        if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
            raise ValueError('Predicted bbox is reversed or empty after scaling')
        return scaled_box

    def score(self, predictions, references):  # pyright: ignore[reportIncompatibleMethodOverride]
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        details = []
        scores = []
        for pred, ref in zip(predictions, references):
            refer = json.loads(ref) if isinstance(ref, str) else ref
            gt_box = [float(value) for value in refer[self.reference_bbox_key]]
            image_width = float(refer[self.image_width_key])
            image_height = float(refer[self.image_height_key])

            detail = {
                'pred': pred,
                'answer': ref,
                'correct': False,
                'coord_mode': f'0-{int(self.coord_scale)}',
            }

            try:
                pred_box_pixel = self._scale_prediction(pred, image_width, image_height)
                iou = _compute_iou(pred_box_pixel, gt_box)
                correct = iou >= self.iou_threshold
                detail['correct'] = correct
                detail['iou'] = iou
                detail['pred_bbox_pixel'] = pred_box_pixel
                scores.append(1 if correct else 0)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                detail['iou'] = 0.0
                detail['pred_bbox_pixel'] = None
                detail['invalid'] = True
                detail['error'] = str(error)
                scores.append(0)

            details.append(detail)

        return {
            f'{self.metric_prefix}@{self.iou_threshold}': 100 * sum(scores) / len(scores) if scores else 0.0,
            'details': details,
        }