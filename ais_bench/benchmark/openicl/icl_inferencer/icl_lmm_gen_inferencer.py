'''
Author: SJTUyh yh_silence@alumni.sjtu.edu.cn
Date: 2026-02-11 16:38:01
LastEditors: SJTUyh yh_silence@alumni.sjtu.edu.cn
LastEditTime: 2026-02-12 18:39:02
FilePath: \benchmark\ais_bench\benchmark\openicl\icl_inferencer\icl_lmm_gen_inferencer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import uuid
from typing import List, Optional

from ais_bench.benchmark.models.output import LMMOutput
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.lmm_gen_inferencer_output_handler import LMMGenInferencerOutputHandler


@ICL_INFERENCERS.register_module()
class LMMGenInferencer(GenInferencer):
    def __init__(
        self,
        model_cfg,
        stopping_criteria: List[str] = [],
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            stopping_criteria=stopping_criteria,
            batch_size=batch_size,
            mode=mode,
            gen_field_replace_token=gen_field_replace_token,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

        self.output_handler = LMMGenInferencerOutputHandler(perf_mode=self.perf_mode,
                                                        save_every=self.save_every)
    def inference(self, retriever: BaseRetriever, output_json_filepath: Optional[str] = None) -> List:
        self.output_handler.set_output_path(output_json_filepath)
        return super().inference(retriever, output_json_filepath)

    def batch_inference(
        self,
        datum,
    ) -> None:
        """Perform batch inference on the given dataloader.

        Args:
            dataloader: DataLoader containing the inference data

        Returns:
            List of inference results
        """
        indexs = datum.pop("index")
        inputs = datum.pop("prompt")
        data_abbrs = datum.pop("data_abbr")
        outputs = [LMMOutput(self.perf_mode) for _ in range(len(indexs))]
        for output in outputs:
            output.uuid = str(uuid.uuid4()).replace("-", "")
        golds = datum.pop("gold", [None] * len(inputs))
        self.model.generate(inputs, outputs, **datum)

        for index, input, output, data_abbr, gold in zip(
            indexs, inputs, outputs, data_abbrs, golds
        ):
            self.output_handler.report_cache_info_sync(
                index, input, output, data_abbr, gold
            )