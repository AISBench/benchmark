"""OneIG 风格评估器 - CSD + SE Encoder 模式"""
import os
import shutil
import tempfile

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
    split_image_grid,
    rm_error,
    ensure_oneig_path,
)


@ICL_EVALUATORS.register_module()
class OneIGStyleEvaluator(BaseEvaluator):
    """
    风格评估器 - CSD + SE Encoder 模式

    提取图片的风格嵌入，与参考风格嵌入计算相似度。

    Args:
        device: str - 运行设备（'cuda' 或 'cpu'）
        csd_embed_path: str - CSD参考嵌入文件路径（.pt文件）
        se_embed_path: str - SE参考嵌入文件路径（.pt文件）
        encoder_cfg: dict - 编码器配置
            - csd_model_path: CSD编码器模型路径
            - clip_model_path: CLIP模型路径
            - se_model_path: SE编码器模型路径

    延迟加载机制：
        - 模型在首次调用score()时加载，不在__init__中加载
    """

    def __init__(self, oneig_root='', device="cuda", csd_embed_path=None, se_embed_path=None, encoder_cfg=None, **kwargs):
        super().__init__()
        self.oneig_root = oneig_root
        self.device = device
        self.csd_embed_path = csd_embed_path
        self.se_embed_path = se_embed_path
        self.encoder_cfg = encoder_cfg or {}
        self.csd_encoder = None
        self.se_encoder = None
        self.csd_ref_embeds = None
        self.se_ref_embeds = None

    def _ensure_models(self):
        """延迟加载CSD和SE编码器以及参考嵌入"""
        if self.csd_encoder is None or self.se_encoder is None:
            try:
                import torch
                import torchvision.transforms as transforms
                csd_model_path = self.encoder_cfg.get(
                    'csd_model_path',
                    os.path.join(self.oneig_root, "scripts", "style", "models", "checkpoint.pth")
                )
                clip_model_path = self.encoder_cfg.get(
                    'clip_model_path',
                    os.path.join(self.oneig_root, "scripts", "style", "models", "ViT-L-14.pt")
                )
                se_model_path = self.encoder_cfg.get('se_model_path', '')

                self.logger.info(f"[Style] Loading encoders...")
                self.logger.info(f"[Style] CSD model path: {csd_model_path}")
                self.logger.info(f"[Style] CLIP model path: {clip_model_path}")
                self.logger.info(f"[Style] SE model path: {se_model_path}")
                self.logger.info(f"[Style] CSD embed path: {self.csd_embed_path}")
                self.logger.info(f"[Style] SE embed path: {self.se_embed_path}")

                ensure_oneig_path(self.oneig_root)
                from scripts.utils.inference import CSDStyleEmbedding, SEStyleEmbedding
                from scripts.utils.CSD_config import CSD_CLIP, convert_state_dict

                # CSD_CLIP 内部使用 clip.load() 加载 CLIP backbone，
                # 默认使用硬编码相对路径 "scripts/style/models/ViT-L-14.pt"。
                # 由于 AISBench 的 CWD 不是 ONEIG_ROOT，需要手动加载。
                self.logger.info(f"[Style] Initializing CSD encoder...")
                model = CSD_CLIP("vit_large", "default", model_path=clip_model_path)
                checkpoint = torch.load(csd_model_path, map_location="cpu", weights_only=False)
                state_dict = convert_state_dict(checkpoint['model_state_dict'])
                model.load_state_dict(state_dict, strict=False)
                model = model.to(self.device)
                # 构造与 CSDStyleEmbedding 兼容的对象
                self.csd_encoder = CSDStyleEmbedding.__new__(CSDStyleEmbedding)
                self.csd_encoder.device = torch.device(self.device)
                self.csd_encoder.model = model
                self.csd_encoder.preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])
                self.logger.info(f"[Style] CSD encoder initialized")

                self.logger.info(f"[Style] Initializing SE encoder...")
                self.se_encoder = SEStyleEmbedding(
                    pretrained_path=se_model_path,
                    device=self.device
                )
                self.logger.info(f"[Style] SE encoder initialized")

                self.logger.info(f"[Style] Loading reference embeddings...")
                if self.csd_embed_path and os.path.exists(self.csd_embed_path):
                    self.csd_ref_embeds = torch.load(self.csd_embed_path, map_location=self.device, weights_only=False)
                    self.logger.info(f"[Style] CSD ref embeds loaded, keys: {list(self.csd_ref_embeds.keys())[:5]}...")
                else:
                    self.logger.warning(f"[Style] CSD embed path not found: {self.csd_embed_path}")

                if self.se_embed_path and os.path.exists(self.se_embed_path):
                    self.se_ref_embeds = torch.load(self.se_embed_path, map_location=self.device, weights_only=False)
                    self.logger.info(f"[Style] SE ref embeds loaded, keys: {list(self.se_ref_embeds.keys())[:5]}...")
                else:
                    self.logger.warning(f"[Style] SE embed path not found: {self.se_embed_path}")

                self.logger.info("Style encoders and reference embeddings loaded successfully")
            except ImportError as e:
                self.logger.error(UTILS_CODES.DEPENDENCY_MODULE_IMPORT_ERROR, f"Failed to import style encoders: {e}")
                raise
            except Exception as e:
                self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"Failed to initialize style encoders: {e}")
                raise

    def score(self, predictions, references, test_set=None, **kwargs):
        if test_set is None:
            self.logger.error(UTILS_CODES.UNKNOWN_ERROR, "test_set is required for OneIGStyleEvaluator")
            return {'accuracy': 0.0, 'details': []}

        self._ensure_models()

        cache_dir = tempfile.mkdtemp()
        self.logger.info(f"[Style] Created cache dir: {cache_dir}")

        style_list = ['abstract_expressionism', 'art_nouveau', 'baroque', 'chinese_ink_painting',
                      'cubism', 'fauvism', 'impressionism', 'line_art', 'minimalism', 'pointillism',
                      'pop_art', 'rococo', 'ukiyo-e', 'clay', 'crayon', 'graffiti', 'lego', 'comic',
                      'pencil_sketch', 'stone_sculpture', 'watercolor', 'celluloid', 'chibi',
                      'cyberpunk', 'ghibli', 'impasto', 'pixar', 'pixel_art', '3d_rendering']

        style_dict = {style: [] for style in style_list}
        prompt_scores = {}
        trace_data = {}  # 落盘溯源数据：sample_id -> dict

        self.logger.info(f"[Style] Starting evaluation, {len(test_set)} samples")

        processed_count = 0
        skipped_count = 0

        try:
            for i, item in enumerate(test_set):
                sample_id = item.get('id', f'sample_{i}')
                image_path = item.get('image_path', '')
                style_label = item.get('style_label', '')
                grid_rows = item.get('grid_rows', 2)
                grid_cols = item.get('grid_cols', 2)

                self.logger.debug(f"[Style] Processing sample {i}: id={sample_id}, style_label={style_label}")

                if style_label:
                    style_label = style_label.lower().replace(' ', '_')
                else:
                    self.logger.warning(f"[Style] No style label for {sample_id}, skipping")
                    skipped_count += 1
                    continue

                if not os.path.exists(image_path):
                    self.logger.warning(f"[Style] Image not found: {image_path}, skipping")
                    skipped_count += 1
                    continue

                try:
                    split_img_list = split_image_grid(
                        image_path,
                        (grid_rows, grid_cols),
                        cache_dir
                    )
                    self.logger.debug(f"[Style] Split {sample_id} into {len(split_img_list)} images")

                    score = []
                    encoder_details = []  # 落盘溯源：逐切分图编码器详情
                    for split_idx, split_img in enumerate(split_img_list):
                        self.logger.debug(f"[Style] Computing embedding for split {split_idx}...")
                        csd_embed = self.csd_encoder.get_style_embedding(split_img)
                        se_embed = self.se_encoder.get_style_embedding(split_img)

                        self.logger.debug(f"[Style] Computing similarity for style: {style_label}")
                        csd_sim = self._compute_similarity(csd_embed, style_label, 'csd')
                        se_sim = self._compute_similarity(se_embed, style_label, 'se')

                        self.logger.debug(f"[Style] Split {split_idx}: CSD_sim={csd_sim:.4f}, SE_sim={se_sim:.4f}")

                        max_style_score = (csd_sim + se_sim) / 2
                        score.append(max_style_score)

                        encoder_details.append({
                            'grid_idx': split_idx,
                            'csd_sim': csd_sim,
                            'se_sim': se_sim,
                            'style_score': max_style_score,
                        })

                    if len(score) != 0:
                        avg_score = sum(score) / len(score)
                        prompt_scores[sample_id] = avg_score
                        self.logger.info(f"[Style] Sample {sample_id} avg_score: {avg_score:.4f}")
                        if style_label in style_dict:
                            style_dict[style_label].append(avg_score)
                        processed_count += 1

                        # 落盘溯源：保存中间数据
                        trace_data[sample_id] = {
                            'image_path': image_path,
                            'grid': f"{grid_rows}x{grid_cols}",
                            'num_splits': len(split_img_list),
                            'style_label': style_label,
                            'encoder_details': encoder_details,
                        }
                    else:
                        prompt_scores[sample_id] = None
                        self.logger.warning(f"[Style] No valid scores for {sample_id}")

                except Exception as e:
                    self.logger.error(UTILS_CODES.UNKNOWN_ERROR, f"[Style] Error processing {sample_id}: {e}")
                    prompt_scores[sample_id] = None
                    skipped_count += 1

            self.logger.info(f"[Style] Processed: {processed_count}, Skipped: {skipped_count}")
        finally:
            shutil.rmtree(cache_dir, onerror=rm_error)

        valid_scores = [s for s in prompt_scores.values() if s is not None]
        overall_score = (
            sum(valid_scores) / len(valid_scores) * 100
            if valid_scores else 0.0
        )

        self.logger.info(f"[Style] Evaluation complete, accuracy={overall_score:.2f}, valid_samples={len(valid_scores)}")

        results = []
        for sample_id, score in prompt_scores.items():
            if score is not None:
                # 从 test_set 中查找对应的 style_label
                style_label = ''
                for item in test_set:
                    if item.get('id') == sample_id:
                        style_label = item.get('style_label', '')
                        break
                results.append({
                    'id': sample_id,
                    'style_label': style_label,
                    'score': score,
                    **trace_data.get(sample_id, {})
                })

        return {
            'accuracy': overall_score,
            'details': results,
            'style_scores': style_dict
        }

    def _compute_similarity(self, embed, style_label, embed_type):
        """计算嵌入与风格的相似度"""
        import torch
        ref_embeds = self.csd_ref_embeds if embed_type == 'csd' else self.se_ref_embeds

        if ref_embeds is None or style_label == '':
            self.logger.info(f"[Style] {embed_type}: ref_embeds is None or style_label empty")
            return 0.0

        if style_label in ref_embeds:
            ref_embed = ref_embeds[style_label]
            if isinstance(embed, torch.Tensor) and isinstance(ref_embed, torch.Tensor):
                if embed.dim() == 1:
                    embed = embed.unsqueeze(0)

                sim = torch.max(embed @ ref_embed.T).item()
                self.logger.debug(f"[Style] {embed_type}: style='{style_label}', similarity={sim:.4f}")
                return max(sim, 0.0)
        else:
            self.logger.warning(f"[Style] {embed_type}: style '{style_label}' NOT found in ref_embeds")

        return 0.0
