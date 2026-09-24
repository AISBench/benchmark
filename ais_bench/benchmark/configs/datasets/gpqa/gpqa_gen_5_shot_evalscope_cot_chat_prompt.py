from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

# ============================================================
# 5-shot CoT prompt
# 参考来源：EvalScope
# ============================================================

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # ==================== 示例 1：Biology — Hardy-Weinberg 遗传平衡 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "In a given population, 1 out of every 400 people has a cancer caused "
                    "by a completely recessive allele, b. Assuming the population is in "
                    "Hardy-Weinberg equilibrium, which of the following is the expected "
                    "proportion of individuals who carry the b allele but are not expected "
                    "to develop the cancer?\n\n"
                    "A) 1/400\n"
                    "B) 19/400\n"
                    "C) 20/400\n"
                    "D) 38/400"
                )),
                dict(role='BOT', prompt=(
                    "The expected proportion of individuals who carry the b allele but "
                    "are not expected to develop the cancer equals the frequency of "
                    "heterozygous individuals in the given population.\n\n"
                    "According to the Hardy-Weinberg equation:\n"
                    "p^2 + 2pq + q^2 = 1\n"
                    "where p = frequency of dominant allele,\n"
                    "      q = frequency of recessive allele,\n"
                    "      q^2 = frequency of homozygous recessive (affected).\n\n"
                    "Given: q^2 = 1/400\n"
                    "Therefore: q = sqrt(1/400) = 1/20 = 0.05\n"
                    "           p = 1 - q = 1 - 0.05 = 0.95\n\n"
                    "The frequency of heterozygous carriers is:\n"
                    "2pq = 2 * 0.95 * 0.05 = 0.095 = 38/400\n\n"
                    "Answer: D"
                )),

                # ==================== 示例 2：Chemistry — 氧化还原滴定双等当点 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "A Fe pellet of 0.056 g is first dissolved in 10 mL of hydrobromic "
                    "acid HBr (0.1 M). The resulting solution is then titrated by KMnO4 "
                    "(0.02 M). How many equivalence points are there?\n\n"
                    "A) Two points, 25 ml and 35 ml\n"
                    "B) One point, 25 mL\n"
                    "C) One point, 10 ml\n"
                    "D) Two points, 25 ml and 30 ml"
                )),
                dict(role='BOT', prompt=(
                    "HBr reacts with Fe to produce Fe2+. In the titration, MnO4- will "
                    "first react with Fe2+, then with Br-.\n\n"
                    "To determine the reaction order, consider the reduction potentials:\n"
                    "  E0(Br2/Br-) = 1.09 V\n"
                    "  E0(MnO4-/Mn2+) = 1.49 V\n"
                    "  E0(Fe3+/Fe2+) = 0.77 V\n\n"
                    "[Fe2+] = m/(M*V) = 0.056/(55.85*0.01) = 0.1 M\n\n"
                    "Reaction 1 (first equivalence point):\n"
                    "  MnO4- + 5Fe2+ + 8H+ -> Mn2+ + 5Fe3+ + 4H2O\n"
                    "  Stoichiometry 1:5, so V_eq1 = (0.1*0.01)/(5*0.02) = 10 mL\n\n"
                    "Reaction 2 (second equivalence point):\n"
                    "  2MnO4- + 10Br- + 16H+ -> 2Mn2+ + 5Br2 + 8H2O\n"
                    "  Stoichiometry 2:10, additional volume = 25 mL\n"
                    "  V_eq2 = 10 + 25 = 35 mL\n\n"
                    "Two equivalence points exist: at 25 mL and 35 mL total added.\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 3：Physics — 三维谐振子能级 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "Consider a quantum mechanical system containing a particle of mass m "
                    "moving in an isotropic three dimensional potential of the form "
                    "V(r) = (1/2)m*omega^2*r^2 corresponding to the force obeying Hooke's "
                    "law. What is the value of the energy of the third excited state, and "
                    "how many linearly independent eigenfunctions are possible for the same "
                    "energy eigenvalue?\n\n"
                    "A) 11*pi^2*hbar^2/(2mr^2), 3\n"
                    "B) (9/2)*hbar*omega, 10\n"
                    "C) 11*pi^2*hbar^2/(2mr^2), 10\n"
                    "D) (9/2)*hbar*omega, 3"
                )),
                dict(role='BOT', prompt=(
                    "This problem is the three-dimensional simple harmonic oscillator (SHO).\n\n"
                    "The energy spectrum of the 3D SHO is:\n"
                    "E_n = (n + 3/2) * hbar * omega,  where n = 0,1,2,3,...\n\n"
                    "For the third excited state, n = 3:\n"
                    "E_3 = (3 + 3/2) * hbar * omega\n"
                    "    = (6/2 + 3/2) * hbar * omega\n"
                    "    = (9/2) * hbar * omega\n\n"
                    "The degeneracy (number of linearly independent eigenfunctions) for "
                    "the 3D isotropic SHO is:\n"
                    "g_n = (n+1)(n+2)/2\n\n"
                    "For n = 3:\n"
                    "g_3 = (3+1)*(3+2)/2 = 4*5/2 = 10\n\n"
                    "Answer: B"
                )),

                # ==================== 示例 4：Chemistry — 色谱极性推断 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "You overhear two chemists talking to each other as they leave a "
                    "synthetic organic chemistry lab. One asks the other \"So, how did it "
                    "go?\" The second chemist replies, \"Not well - my compounds are on "
                    "top of each other.\" What is the second chemist most likely "
                    "referring to?\n\n"
                    "A) The compounds they are working with have similar polarities.\n"
                    "B) The compounds they are working with have similar boiling points.\n"
                    "C) The compounds they are working with are bonding to each other "
                    "through non-covalent/van der Waals interactions.\n"
                    "D) The compounds they are working with have similar optical rotations."
                )),
                dict(role='BOT', prompt=(
                    "\"On top of each other\" is common lab jargon referring to compounds "
                    "that have similar Rf values on thin-layer chromatography (TLC), a "
                    "routine operation in synthetic organic chemistry.\n\n"
                    "On a TLC plate, compounds with the same Rf value appear at the same "
                    "vertical position — literally \"on top of each other\" — making "
                    "separation and identification difficult.\n\n"
                    "Similar Rf values arise primarily from compounds having similar "
                    "polarities, as polarity determines the interaction strength with "
                    "the stationary phase (silica gel) versus the mobile phase (solvent).\n\n"
                    "Answer: A"
                )),

                # ==================== 示例 5：Math — 无穷几何级数求概率 ====================
                dict(role='HUMAN', prompt=(
                    "Answer the following multiple choice question. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                    "Think step by step before answering.\n\n"
                    "Two people are playing the following game. A fair coin is tossed "
                    "repeatedly. Person A wins if a tail appears on an odd-numbered toss "
                    "(1st, 3rd, 5th, ...). Person B wins if a tail appears on an "
                    "even-numbered toss (2nd, 4th, 6th, ...). Imagine this game played "
                    "forever. What is the probability that person A wins this game?\n\n"
                    "A) 1/2\n"
                    "B) 1/4\n"
                    "C) 2/3\n"
                    "D) 1/8"
                )),
                dict(role='BOT', prompt=(
                    "Person A wins if a tail appears on the 1st toss: P = 1/2.\n"
                    "If the 1st toss is heads AND 2nd is heads (otherwise B wins on 2nd), "
                    "A can win on the 3rd toss: P = (1/2)*(1/2)*(1/2) = (1/2)^3 = 1/8.\n"
                    "Similarly, A wins on the 5th toss: P = (1/2)^5 = 1/32.\n"
                    "And so on for the 7th, 9th, 11th...\n\n"
                    "This forms an infinite geometric series:\n"
                    "P(A wins) = (1/2) + (1/2)^3 + (1/2)^5 + (1/2)^7 + ...\n\n"
                    "For an infinite geometric series: S = a1 / (1 - r)\n"
                    "where a1 = first term = 1/2,\n"
                    "      r  = common ratio = (1/2)^2 = 1/4.\n\n"
                    "P(A wins) = (1/2) / (1 - 1/4)\n"
                    "         = (1/2) / (3/4)\n"
                    "         = (1/2) * (4/3)\n"
                    "         = 2/3\n\n"
                    "Answer: C"
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
