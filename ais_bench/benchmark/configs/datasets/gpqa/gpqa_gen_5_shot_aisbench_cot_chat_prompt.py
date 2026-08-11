from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

# ============================================================
# 5-shot CoT prompt
# 选取原则：
#   1. 全部来自 gpqa_extended.csv，不在 diamond 中（防止数据泄露）
#   2. 推理链自洽
#   3. 步骤显式展开
#   4. 领域覆盖：Physics×2 + Chemistry×2 + Biology×1
#   5. 推理模式多样
# ============================================================

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # ==================== 示例 1：Physics — 绝热定理 + 能级计算 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "A particle is placed in an infinite, one-dimensional potential well "
                    "with boundaries at x=0 and x=a. If the wall is moved slowly from x=a "
                    "to x=4a, calculate the work done in this process to move the wall in "
                    "terms of E. (E = pi^2*hbar^2/(2ma^2))\n\n"
                    "A) -(15/16)E\n"
                    "B) (63/128)E\n"
                    "C) -(63/64)E\n"
                    "D) -(7/16)E"
                )),
                dict(role='BOT', prompt=(
                    "Since the wall is moved slowly, the adiabatic theorem says that the "
                    "particle will make no transition. So it will be found in the ground "
                    "state of the new potential well with boundaries x=0 to x=4a.\n\n"
                    "The work done is:\n"
                    "W = E_ground(new) - E_ground(old)\n"
                    "  = pi^2*hbar^2/(2m(4a)^2) - pi^2*hbar^2/(2ma^2)\n"
                    "  = pi^2*hbar^2/(32ma^2) - pi^2*hbar^2/(2ma^2)\n"
                    "  = -(15/32) * pi^2*hbar^2/(ma^2)\n"
                    "  = -(15/16)E\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 2：Physics — 相对论粒子束衰减 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "A collimated beam emerges from an analyzing spectrometer with E=4 GeV. "
                    "The beam is composed of particle X, with m_X=1.48 GeV and "
                    "tau_0=8.7x10^-5 s. At what distance is the flux reduced to 25%?\n\n"
                    "A) 90954 m\n"
                    "B) 17586 m\n"
                    "C) 48635 m\n"
                    "D) 64218 m"
                )),
                dict(role='BOT', prompt=(
                    "We have gamma = E/m_X = 4/1.48 = 2.702, and beta = sqrt(1-1/gamma^2) = 0.93.\n\n"
                    "The laboratory lifetime is:\n"
                    "tau = gamma * tau_0 = 2.702 * 8.7x10^-5 = 2.35x10^-4 s\n\n"
                    "The time needed to reduce the flux from I_0 to I_0/4:\n"
                    "I_0 * e^(-t/tau) = I_0/4\n"
                    "e^(-t/tau) = 1/4\n"
                    "t = tau * ln(4) = 2.35x10^-4 * 1.386 = 3.26x10^-4 s\n\n"
                    "Finally, the distance traveled during time t:\n"
                    "d = t * beta * c = 3.26x10^-4 * 0.93 * 3x10^8 = 90954 m\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 3：Chemistry — pH 缓冲溶液计算 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "Calculate the pH of a mixture of 300 mL of 0.2 M HCN, 200 mL of "
                    "0.2 M HCl, and 350 mL of 0.2 M NaOH. Given that Ka(HCN) = 1.8x10^-5.\n\n"
                    "A) 4.74\n"
                    "B) 8.78\n"
                    "C) 5.25\n"
                    "D) 1.33"
                )),
                dict(role='BOT', prompt=(
                    "First, calculate the amount of each species in mmol:\n"
                    "HCN: 300 mL * 0.2 M = 60 mmol\n"
                    "HCl:  200 mL * 0.2 M = 40 mmol\n"
                    "NaOH: 350 mL * 0.2 M = 70 mmol\n\n"
                    "Step 1: HCl (strong acid) reacts with NaOH first.\n"
                    "40 mmol HCl + 40 mmol NaOH -> 40 mmol NaCl + 40 mmol H2O\n"
                    "Remaining NaOH: 70 - 40 = 30 mmol\n\n"
                    "Step 2: Remaining NaOH reacts with HCN (weak acid).\n"
                    "30 mmol NaOH + 30 mmol HCN -> 30 mmol NaCN + 30 mmol H2O\n"
                    "Remaining HCN: 60 - 30 = 30 mmol\n"
                    "NaCN produced: 30 mmol\n\n"
                    "Step 3: The solution now contains 30 mmol HCN (weak acid) and "
                    "30 mmol NaCN (its conjugate base) — this is a buffer.\n\n"
                    "pH = pKa + log([salt]/[acid])\n"
                    "pKa = -log(1.8x10^-5) = 4.74\n"
                    "pH = 4.74 + log(30/30) = 4.74 + 0 = 4.74\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 4：Chemistry — 反应产物枚举 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "m-toluidine and hexane-2,4-dione are heated in the presence of acid. "
                    "In the ensuing reaction how many organic products can form?\n\n"
                    "A) 4\n"
                    "B) 1\n"
                    "C) 2\n"
                    "D) 3"
                )),
                dict(role='BOT', prompt=(
                    "This reaction is the Combes Quinoline Synthesis. The amine condenses "
                    "with a ketone to form an imine, which then undergoes electrophilic "
                    "aromatic addition and dehydration to form a substituted quinoline.\n\n"
                    "With m-toluidine (3-methylaniline) and hexane-2,4-dione, the "
                    "unsymmetrical ketone allows for two possible ring-closure positions, "
                    "and the methyl group on the aniline ring can end up at different "
                    "positions relative to the substituents. This gives 4 possible products:\n"
                    "1. 4-ethyl-2,7-dimethylquinoline\n"
                    "2. 2-ethyl-4,7-dimethylquinoline\n"
                    "3. 4-ethyl-2,5-dimethylquinoline\n"
                    "4. 2-ethyl-4,5-dimethylquinoline\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 5：Biology — NNK 密码子文库多样性 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "You have a 10 uL aliquot of a 10 uM DNA template of a protein library. "
                    "The template contains 12 NNK codons in the coding region. What is the "
                    "order of magnitude of the maximum possible number of unique full-length "
                    "protein sequences that can be translated from the aliquot of DNA?\n\n"
                    "A) 10^13\n"
                    "B) 10^11\n"
                    "C) 10^18\n"
                    "D) 10^15"
                )),
                dict(role='BOT', prompt=(
                    "NNK codons code for all 20 amino acids. Therefore, the theoretical "
                    "maximum protein diversity is:\n"
                    "20^12 = (2x10^1)^12 = 2^12 x 10^12 = 4096 x 10^12 ≈ 4x10^15\n"
                    "This is approximately 10^15.\n\n"
                    "However, this exceeds the number of DNA molecules actually present "
                    "in the aliquot. The number of DNA molecules is:\n"
                    "N = volume * concentration * Avogadro's number\n"
                    "  = (10x10^-6 L) * (10x10^-6 mol/L) * (6x10^23 molecules/mol)\n"
                    "  = 10^-5 * 10^-5 * 6x10^23\n"
                    "  = 6x10^13 ≈ 10^13 molecules\n\n"
                    "Since the diversity is limited by the number of DNA molecules "
                    "(the bottleneck), the maximum possible number of unique sequences "
                    "is approximately 10^13.\n\n"
                    "Answer: A"
                )),

                # ==================== 实际测试题 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "{question}\n\n"
                    "A) {A}\n"
                    "B) {B}\n"
                    "C) {C}\n"
                    "D) {D}"
                )),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

# 严格匹配
# gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
#                      pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess))

# 宽松匹配
gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='ais_bench/datasets/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
