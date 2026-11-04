import logging
from transformers import AutoTokenizer
from data_picker import *
import os
import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset(tokenizer_path: str, input_len: int, number: int, prefix_flag):

    logging.info(f"加载tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    output_samples = []
    attempts = 0
    max_attempts = number * 10  # 防止意外无限循环

    pbar = tqdm(total=number, desc="Generating dataset", unit="row") if tqdm else None
    while len(output_samples) < number and attempts < max_attempts:
        attempts += 1
        # 随机选择一条文本
        picker = DataPicker(os.path.join(os.path.dirname(__file__), "GSM8K.jsonl"), "./picked_ids.txt", prefix_flag)
        raw_text = picker.pick_one()
        if raw_text is None:
            break

        # tokenize
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)

        if len(tokens) == 0:
            # logging.info(f"生成数据集失败，请清空pick ids")
            break

        # 根据需求调整长度：重复或截断
        if len(tokens) >= input_len:
            # 截断到 input_len
            adjusted_tokens = tokens[:input_len]
        else:
            # 重复整个序列直到达到 input_len
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            repeated_tokens = tokens * repeat_times
            adjusted_tokens = repeated_tokens[:input_len]

        # 解码回文本
        adjusted_text = tokenizer.decode(adjusted_tokens, skip_special_tokens=True)
        final_len = len(tokenizer.encode(adjusted_text, add_special_tokens=False))
        if final_len != input_len:
            corrected_tokens = tokenizer.encode(adjusted_text, add_special_tokens=False)
            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            else:
                corrected_tokens = (corrected_tokens * ((input_len // len(corrected_tokens)) + 1))[:input_len]
            adjusted_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)


        output_samples.append(adjusted_text)
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    if len(output_samples) < number:
        return None

    return output_samples


def generate_unique_tokens(tokenizer_path, seed, n, number):
    """
    根据模型 tokenizer 和随机种子，生成 n 个不相同的 token，共 number 行数据

    Args:
        model_name_or_tokenizer: 模型名称或已加载的 tokenizer 对象
        seed: 随机种子
        n: 每行需要生成的 token 数量
        number: 需要生成的数据行数

    Returns:
        list: 包含 number 行数据的列表
    """
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 获取词表大小
    vocab_size = len(tokenizer)

    if n > vocab_size:
        raise ValueError(f"每行请求的 token 数量 {n} 超过词表大小 {vocab_size}")

    all_lines = []
    pbar = tqdm(total=number, desc="Generating unique tokens", unit="row") if tqdm else None

    for line_idx in range(number):
        if pbar:
            pbar.update(1)
        # 为每一行使用不同的种子，确保行间数据不重复
        line_seed = seed + line_idx
        random.seed(line_seed)
        torch.manual_seed(line_seed)

        unique_tokens = []
        seen_tokens = set()
        max_attempts = n * 10  # 防止无限循环
        attempts = 0

        while len(unique_tokens) < n and attempts < max_attempts:
            # 随机生成 token ID
            token_id = random.randint(0, vocab_size - 1)

            # 检查是否重复
            if token_id in seen_tokens:
                attempts += 1
                continue

            # 转换为文本
            try:
                token_text = tokenizer.decode([token_id])

                # 可选：跳过特殊 token 或空 token
                # if token_text.strip() or token_text:  # 保留非空 token
                unique_tokens.append(token_text)
                seen_tokens.add(token_id)
            except:
                pass

            attempts += 1

        if len(unique_tokens) < n:
            print(f"警告：第 {line_idx + 1} 行只生成了 {len(unique_tokens)} 个唯一 token")
        # combined_text = ''.join(unique_tokens)
        all_lines.append(''.join(unique_tokens))

    if pbar:
        pbar.close()
    return all_lines

def write_data(path, dataset, num):
    if num is not None:
        if len(dataset) < num:
            # 重复数据
            repeats = num // len(dataset)
            remainder = num % len(dataset)
            dataset = dataset * repeats + dataset[:remainder]
        else:
            # 截取数据
            dataset = dataset[:num]
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False))
            f.write("\n")

def sample_target_length(rng, fixed_length, length_mean=None, length_std=None, length_min=None, length_max=None):
    """从高斯或均匀分布中采样目标长度"""
    fixed_length = max(1, int(fixed_length))
    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))
    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def _build_length_tag(input_len, length_mean, length_std, length_min, length_max):
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(input_len)}"


def _truncate_or_pad_text(tokenizer, text, target_len):
    """将文本的 token 长度调整到 target_len（截断或重复填充）"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        repeat_times = (target_len + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeat_times)[:target_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def create_multi_prefix_dataset(tokenizer_path: str, input_len: int, number: int, save_path, prefix_flag, dp, repeat_rate, seed, prefix_num,
                                length_mean=None, length_std=None, length_min=None, length_max=None):
    random.seed(seed)
    torch.manual_seed(seed)
    Path = __import__("pathlib").Path
    Path("picked_ids.txt").unlink(missing_ok=True)
    base_name = os.path.basename(os.path.normpath(tokenizer_path))
    use_variable_length = (
        (length_mean is not None and length_std is not None)
        or (length_min is not None and length_max is not None)
    )

    # ========== 普通数据集（无前缀）==========
    if prefix_flag == 0:
        if use_variable_length:
            rng = random.Random(seed)
            real_lens = [max(1, int(sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max))) for _ in range(number)]
            max_len = max(real_lens)
            # 先生成统一最大长度的文本池，再逐条截断
            long_texts = create_dataset(tokenizer_path, max_len, number, 0)
            if long_texts is None:
                logging.error("生成数据集失败，请清空picked ids")
                exit(0)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            dataset = []
            pbar = tqdm(total=number, desc="Truncating to variable lengths", unit="row") if tqdm else None
            for i, rl in enumerate(real_lens):
                adjusted = _truncate_or_pad_text(tokenizer, long_texts[i], rl)
                dataset.append(adjusted)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
            length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
            dataset_path = os.path.join(save_path, f'GSM8K-{length_tag}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path
        else:
            dataset = create_dataset(tokenizer_path, input_len, number, 0)
            dataset_path = os.path.join(save_path, f'GSM8K-in{input_len}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path

    # ========== 前缀数据集 ==========
    if use_variable_length:
        return _create_prefix_dataset_variable(tokenizer_path, input_len, number, save_path, base_name, dp, repeat_rate, seed, prefix_num,
                                               length_mean, length_std, length_min, length_max)

    # -------- 定长前缀数据集（原有逻辑）--------
    prefix_len = int(input_len * repeat_rate)
    prefix_data = []
    prefix_data = create_dataset(tokenizer_path, prefix_len, prefix_num, 1)
    if prefix_data == None and repeat_rate > 0:
        logging.error(f"生成数据集失败，请清空picked ids")
        exit(0)

    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])

    prefix_path = os.path.join(save_path, f'prefix-GSM8K-in{prefix_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp*prefix_num)
    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'GSM8K-in{prefix_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # 前缀后插入3个随机token
    uniq_token_set = generate_unique_tokens(tokenizer_path, seed, 3, number)
    # 后缀数据
    suffix_len = int(input_len - prefix_len - 3)
    suffix_dataset = create_dataset(tokenizer_path, suffix_len, number, 0)

    # 拼接完整数据集
    dataset = []
    data_len = 0
    pbar = tqdm(total=number, desc="Stitching dataset", unit="row") if tqdm else None
    while data_len < number:
        single_data = prefix_data[data_len % prefix_num] + uniq_token_set[data_len] + suffix_dataset[data_len]
        dataset.append(single_data)
        data_len += 1
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    dataset_path = os.path.join(save_path, f'GSM8K-in{input_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    return prefix_path, dataset_path


def _create_prefix_dataset_variable(tokenizer_path, input_len, number, save_path, base_name, dp, repeat_rate, seed, prefix_num,
                                    length_mean, length_std, length_min, length_max):
    """变长前缀数据集生成：预采样长度 → 前缀池 → 逐条截断前缀/后缀并拼接"""
    rng = random.Random(seed)

    # 预采样每条数据的实际长度和公共前缀长度
    real_lens = [max(1, int(sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max))) for _ in range(number)]
    common_lens = [max(0, min(rl, int(round(rl * repeat_rate)))) for rl in real_lens]
    max_common_len = max(common_lens) if common_lens else 0

    # 生成前缀池（统一最大公共长度，后续逐条截断）
    prefix_data = []
    if max_common_len > 0:
        prefix_data = create_dataset(tokenizer_path, max_common_len, prefix_num, 1)
        if prefix_data is None:
            logging.error("生成数据集失败，请清空picked ids")
            exit(0)
    else:
        prefix_data = [""] * prefix_num

    # 写前缀文件
    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])
    prefix_path = os.path.join(save_path, f'prefix-GSM8K-in{max_common_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp*prefix_num)
    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'GSM8K-in{max_common_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # 生成唯一 token 集合（每组3个不同token）
    uniq_token_set = generate_unique_tokens(tokenizer_path, seed, 3, number)

    # 后缀池：生成最大后缀长度，逐条截断
    max_suffix_len = max(rl - cl - 3 for rl, cl in zip(real_lens, common_lens))
    if max_suffix_len < 1:
        max_suffix_len = 1
    suffix_pool = create_dataset(tokenizer_path, max_suffix_len, number, 0)
    if suffix_pool is None:
        logging.error("生成后缀数据集失败，请清空picked ids")
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 逐条拼接
    dataset = []
    pbar = tqdm(total=number, desc="Stitching dataset (variable)", unit="row") if tqdm else None
    for idx in range(number):
        rl = real_lens[idx]
        cl = common_lens[idx]
        suffix_len_needed = max(0, rl - cl - 3)

        # 前缀截断
        prefix_text = prefix_data[idx % prefix_num]
        if cl > 0 and prefix_text:
            prefix_text = _truncate_or_pad_text(tokenizer, prefix_text, cl)
        else:
            prefix_text = ""

        # 后缀截断
        suffix_text = suffix_pool[idx]
        if suffix_len_needed > 0 and suffix_text:
            suffix_text = _truncate_or_pad_text(tokenizer, suffix_text, suffix_len_needed)
        else:
            suffix_text = ""

        single_data = prefix_text + uniq_token_set[idx] + suffix_text
        dataset.append(single_data)

        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
    dataset_path = os.path.join(save_path, f'GSM8K-{length_tag}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    logging.info(f"  max_common_len={max_common_len}, max_suffix_len={max_suffix_len}")
    logging.info(f"  avg_hit_ratio={sum(c / r for c, r in zip(common_lens, real_lens)) / len(real_lens):.2%}")

    return prefix_path, dataset_path

def parse_prefix_ratio(r: str) -> float:
    """
    "50%" -> 0.5, "0.5" -> 0.5, "0.500" -> 0.5
    """
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("prefix-ratio 必须在 [0,1] 区间或百分数 [0%,100%]")
    return v
