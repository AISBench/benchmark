from mmengine.config import read_base

with read_base():
    # gsm8k
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_4_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets as gsm8k_4_shot_cot_str

    # ceval
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_cot_chat_prompt import ceval_datasets as ceval_0_shot_chat
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_str import ceval_datasets as ceval_0_shot_str
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets as ceval_5_shot_str

    # drop
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_0_shot_str import drop_datasets as drop_0_shot_str
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_3_shot_str import drop_datasets as drop_3_shot_str

    # gpqa
    from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_str import gpqa_datasets as gpqa_0_shot_str

    # aime2024
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_str import aime2024_datasets as aime2024_0_shot_str

    # humaneval
    from ais_bench.benchmark.configs.datasets.humaneval.humaneval_gen_0_shot import humaneval_datasets as humaneval_0_shot_str

    # math
    from ais_bench.benchmark.configs.datasets.math.math_prm800k_500_0shot_cot_gen import math_datasets as math500_0_shot_str
    from ais_bench.benchmark.configs.datasets.math.math_prm800k_500_5shot_cot_gen import math_datasets as math500_5_shot_str

    # mmlu
    from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_str import mmlu_datasets as mmlu_5_shot_str

    # mmlu_pro
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_0_shot_str import mmlu_pro_datasets as mmlu_pro_0_shot_str
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_5_shot_str import mmlu_pro_datasets as mmlu_pro_5_shot_str

    # boolq
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_0_shot_cot_str import BoolQ_datasets as boolq_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_0_shot_str import BoolQ_datasets as boolq_0_shot_str
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_5_shot_str import BoolQ_datasets as boolq_5_shot_str

    # bbh
    from ais_bench.benchmark.configs.datasets.bbh.bbh_gen_3_shot_cot_chat import bbh_datasets as bbh_3_shot_cot_chat

    # livecodebench
    from ais_bench.benchmark.configs.datasets.livecodebench.livecodebench_code_generate_lite_gen_0_shot_chat import LCB_datasets as LCB_code_gen_lite_0_shot_chat

    # piqa
    from ais_bench.benchmark.configs.datasets.piqa.piqa_gen_0_shot_chat_prompt import piqa_datasets as piqa_0_shot_chat
    from ais_bench.benchmark.configs.datasets.piqa.piqa_gen_0_shot_str import piqa_datasets as piqa_0_shot_str

    # agieval
    from ais_bench.benchmark.configs.datasets.agieval.agieval_gen_0_shot_chat_prompt import agieval_datasets as agieval_0_shot_chat

    # cmmlu
    from ais_bench.benchmark.configs.datasets.cmmlu.cmmlu_gen_0_shot_cot_chat_prompt import cmmlu_datasets as cmmlu_0_shot_chat
    from ais_bench.benchmark.configs.datasets.cmmlu.cmmlu_gen_5_shot_cot_chat_prompt import cmmlu_datasets as cmmlu_5_shot_chat

    # hellaswag
    from ais_bench.benchmark.configs.datasets.hellaswag.hellaswag_gen_0_shot_chat_prompt import hellaswag_datasets as hellaswag_0_shot_chat
    from ais_bench.benchmark.configs.datasets.hellaswag.hellaswag_gen_10_shot_chat_prompt import hellaswag_datasets as hellaswag_10_shot_chat

    # mbpp
    from ais_bench.benchmark.configs.datasets.mbpp.mbpp_passk_gen_3_shot_chat_prompt import mbpp_datasets as mbpp_3_shot_chat
    from ais_bench.benchmark.configs.datasets.mbpp.sanitized_mbpp_passk_gen_3_shot_chat_prompt import sanitized_mbpp_datasets as sanitized_mbpp_3_shot_chat

    # mgsm
    from ais_bench.benchmark.configs.datasets.mgsm.mgsm_gen_0_shot_cot_chat_prompt import mgsm_datasets as mgsm_0_shot_chat
    from ais_bench.benchmark.configs.datasets.mgsm.mgsm_gen_8_shot_cot_chat_prompt import mgsm_datasets as mgsm_8_shot_chat

    # ARC_c
    from ais_bench.benchmark.configs.datasets.ARC_c.ARC_c_gen_0_shot_chat_prompt import ARC_c_datasets as ARC_c_0_shot_chat
    from ais_bench.benchmark.configs.datasets.ARC_c.ARC_c_gen_25_shot_chat_prompt import ARC_c_datasets as ARC_c_25_shot_chat

    # ARC_e
    from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_0_shot_chat_prompt import ARC_e_datasets as ARC_e_0_shot_chat
    from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_25_shot_chat_prompt import ARC_e_datasets as ARC_e_25_shot_chat

    # race
    ## race-middle
    from ais_bench.benchmark.configs.datasets.race.race_middle_gen_5_shot_chat import race_datasets as race_middle_5_shot_chat
    from ais_bench.benchmark.configs.datasets.race.race_middle_gen_5_shot_cot_chat import race_datasets as race_middle_5_shot_cot_chat
    ## race-high
    from ais_bench.benchmark.configs.datasets.race.race_high_gen_5_shot_chat import race_datasets as race_high_5_shot_chat
    from ais_bench.benchmark.configs.datasets.race.race_high_gen_5_shot_cot_chat import race_datasets as race_high_5_shot_cot_chat

    # triviaqa
    from ais_bench.benchmark.configs.datasets.triviaqa.triviaqa_gen_5_shot_chat_prompt import triviaqa_datasets as triviaqa_5_shot_chat

    # winogrande
    from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_0_shot_chat_prompt import winogrande_datasets as winogrande_0_shot_chat
    from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_5_shot_chat_prompt import winogrande_datasets as winogrande_5_shot_chat

    # humanevalx
    from ais_bench.benchmark.configs.datasets.humanevalx.humanevalx_gen_0_shot  import humanevalx_datasets as humanevalx_0_shot_str

    # ifeval
    from ais_bench.benchmark.configs.datasets.ifeval.ifeval_0_shot_gen_str import ifeval_datasets as ifeval_0_shot_gen_str
