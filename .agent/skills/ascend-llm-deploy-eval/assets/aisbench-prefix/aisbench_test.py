import json as _event_json, os as _event_os
_event_stage = None
def _benchmark_event(stage, status):
    global _event_stage
    _event_stage = stage if status == "doing" else None
    if _event_os.environ.get("BENCH_EVENT_ROOT"):
        from task_events import emit
        emit(_event_os.environ["BENCH_EVENT_ROOT"], stage, status, **_event_json.loads(_event_os.environ["BENCH_EVENT_CONTEXT"]))
def _benchmark_run(*args, **kwargs):
    try:
        result = subprocess.run(*args, **kwargs)
        from log_format import normalize_log
        normalize_log("aisbench.log")
        return result
    except Exception:
        if _event_stage:
            _benchmark_event(_event_stage, "fail")
        raise
import os, errno
import sys
import argparse
import re
import logging
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from runtime_config import *
from generate_dataset import *
from save_file import get_data, save_csv, save_log
from cal_prefix_hit_rate import *
logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=str, default="1500", help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=str, default="2048", help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="0", help="request rate")
    parser.add_argument("--test_type", type=str, default="stream", help="text or stream")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--enable_think", action='store_true', default=False, help="enable thinking for ds v3.1")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp", type=int, default=1, help="dp size")
    parser.add_argument("--length_mean", type=int, default=None, help="gaussian mean for variable length")
    parser.add_argument("--length_std", type=float, default=None, help="gaussian std for variable length")
    parser.add_argument("--length_min", type=int, default=None, help="min length for uniform range or gaussian clip")
    parser.add_argument("--length_max", type=int, default=None, help="max length for uniform range or gaussian clip")
    args = parser.parse_args()
    if os.environ.get('BENCH_CASE_CONFIG'):
        from runtime_config import case_arguments
        vars(args).update(case_arguments())
    return args

def symlink_force(target, link_name):
    logging.info(f"make symlink: {link_name} ==> {target}")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
    if os.environ.get('BENCH_LINK_MANIFEST'):
        from tool_links import record
        record(link_name)


def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, prefix_num, repeat_rate, seed,
                         length_mean=None, length_std=None, length_min=None, length_max=None):
    if not os.path.exists(dataset_path):
        logging.warning(f"dataset work path {dataset_path} not exist. creating it automatically.")
        os.makedirs(dataset_path, exist_ok=True)
        logging.info(f"dataset work path {dataset_path} created.")

    base_name = os.path.basename(os.path.normpath(model_path))
    if dataset_type == "prefix_cache":
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num,
                                                                             length_mean, length_std, length_min, length_max)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={dp*prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
        if length_mean is not None and length_std is not None:
            logging.info(f"  length_dist=gaussian(mean={length_mean}, std={length_std}, min={length_min}, max={length_max})")
        elif length_min is not None and length_max is not None:
            logging.info(f"  length_dist=uniform_int([{length_min}, {length_max}])")
        else:
            logging.info("  length_dist=fixed")
    else:
        dataset_name = "GSM8K-in" + str(input_len) + "-num" + str(data_num) + "-" + base_name + ".jsonl"
        logging.info(f"dataset_name: {dataset_name}")
        dataset_jsonl_path = os.path.join(dataset_path, dataset_name)
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not os.path.exists(dataset_jsonl_path):
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num,
                                                                                 length_mean, length_std, length_min, length_max)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path

def generate_aisbench_command(DEFAULT_PERFORMANCE_TEST):
    if test_accuracy:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --work-dir {OUTPUT_DIR} --dump-eval-details"
    else:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer {DEFAULT_PERFORMANCE_TEST} --work-dir {OUTPUT_DIR} --debug --num-warmups 0 >> aisbench.log 2>&1"
    return ais_bench_cmd

def generate_test_dataset(src_file, dst_dir):
    dst_file = os.path.join(dst_dir, "test.jsonl")
    logging.info(f"src_file: {src_file}")
    logging.info(f"dst_file: {dst_file}")
    # 使用软连接
    symlink_force(src_file, dst_file)
    return

def save_result(request_rate, npu_num, repeat_rate):
    aisbench_log_dir = "aisbench.log"
    filename = "aisbench_result.csv"
    ans, log_dir=get_data(aisbench_log_dir,request_rate,npu_num,repeat_rate)
    save_csv(ans, filename)

def modify_aisbench_api(concurrency, output_len, request_rate_param=None):
    if request_rate_param is None:
        request_rate_param = request_rate
    file_default = open(os.path.join(os.environ.get("BENCH_TOOL_SOURCE", os.path.dirname(__file__)), "default_api.py"), "r")
    file_temp = open("temp_api.py", 'w+')
    logging.info("Api config file:")
    for ss in file_default.readlines():
        tt = re.sub("model_path_for_replace", MODEL_PATH, ss)
        tt = re.sub("model_name_for_replace", MODEL_NAME, tt)
        tt = re.sub("api_key_for_replace", API_KEY, tt)
        tt = re.sub("rr_for_replace", request_rate_param, tt)
        tt = re.sub("test_type_for_replace", api_test_type, tt)
        tt = re.sub("test_abbr_for_replace", api_test_abbr, tt)
        tt = re.sub("ip_for_replace", HOST_IP, tt)
        tt = re.sub("port_for_replace", HOST_PORT, tt)
        tt = re.sub("url_for_replace", URL, tt)
        tt = re.sub("outputlen_for_replace", output_len, tt)
        tt = re.sub("concurrency_for_replace", concurrency, tt)
        if test_accuracy:
            generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
        else:
            generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"
        if enable_think:
            generation_kwargs = generation_kwargs + ",\n\t\t\tchat_template_kwargs={\"enable_thinking\": True}"
        tt = re.sub("generation_kwargs_for_replace", generation_kwargs.expandtabs(4), tt)
        print(tt, end='')
        file_temp.write(tt)
    file_default.close()
    file_temp.close()
    symlink_force(
        os.path.join(os.getcwd(), "temp_api.py"),
        os.path.join(WORK_PATH, "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py")
    )

def _fetch_pod_metrics(pod):
    if pod.startswith('['):
        # IPv6 格式: [ip]:port
        ip = pod.split(']:')[0][1:]
        port = pod.split(']:')[1]
    elif pod.count(':') > 1:
        # IPv6 格式无方括号: ip:port
        ip, port = pod.rsplit(':', 1)
    else:
        # IPv4 格式: ip:port
        ip, port = pod.split(":")
    query_token, query_external = get_prefix_queries_total(ip, port)
    hit_token, hit_external = get_prefix_hits_total(ip, port)
    return pod, query_token, query_external, hit_token, hit_external

def get_pod_metrics_info(pod_info):
    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=len(pod_info)) as executor:
        futures = [executor.submit(_fetch_pod_metrics, pod) for pod in pod_info]
        for future in as_completed(futures):
            pod, q_n, q_e, h_n, h_e = future.result()
            query_tokens[pod] = q_n
            query_tokens_external[pod] = q_e
            hit_tokens[pod] = h_n
            hit_tokens_external[pod] = h_e
    import json, time
    with open("metrics-snapshots.jsonl", "a") as snapshot:
        snapshot.write(json.dumps({"time": time.time(), "query": query_tokens, "external_query": query_tokens_external, "hit": hit_tokens, "external_hit": hit_tokens_external}) + "\n")
    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external

def cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                        query_tokens_external_new, hit_tokens_new, hit_tokens_external_new):
    if not query_tokens:
        logging.warning("No prefix cache metrics found.")
        return

    # 输出格式
    line_width = 118
    scope_width, hbm_rate_width, hbm_detail_width, external_rate_width = 35, 19, 21, 21
    headers = ["scope", "hbm_hit_rate", "hbm(hit/query)", "external_hit_rate", "external(hit/query)"]
    header = (f"{headers[0]:<{scope_width}}{headers[1]:<{hbm_rate_width}}"
              f"{headers[2]:<{hbm_detail_width}}{headers[3]:<{external_rate_width}}{headers[4]}")

    def format_rate(hits, queries):
        return "unavailable" if queries <= 0 else format(hits / queries, ".2%")

    def build_row(scope, hbm_hits, hbm_queries, external_hits, external_queries):
        return (f"{scope:<{scope_width}}{format_rate(hbm_hits, hbm_queries):<{hbm_rate_width}}"
                f"{f'{hbm_hits}/{hbm_queries}':<{hbm_detail_width}}"
                f"{format_rate(external_hits, external_queries):<{external_rate_width}}"
                f"{external_hits}/{external_queries}")

    total_hbm_queries = total_hbm_hits = total_external_queries = total_external_hits = 0
    pod_sections, skipped_pods = [], []

    for pod, engines in sorted(query_tokens.items()):
        # 八类指标必须全部存在，且 engine_id 完全一致；
        # 否则无法可靠计算"测试后累计值 - 测试前累计值"，跳过该 POD
        metrics = [query_tokens.get(pod), query_tokens_external.get(pod), hit_tokens.get(pod),
                   hit_tokens_external.get(pod), query_tokens_new.get(pod), query_tokens_external_new.get(pod),
                   hit_tokens_new.get(pod), hit_tokens_external_new.get(pod)]

        if not all(metrics):
            skipped_pods.append(pod)
            continue

        engine_ids = set(engines)
        if any(set(metric) != engine_ids for metric in metrics[1:]):
            skipped_pods.append(pod)
            continue

        # 按 engine_id 升序遍历，为每个 engine 单独生成一行
        engine_rows = []
        for engine_id in sorted(engine_ids):
            engine_scope = f"{pod}/engine{engine_id}"
            engine_hbm_queries = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            engine_hbm_hits = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            engine_external_queries = query_tokens_external_new[pod][engine_id] \
                - query_tokens_external[pod][engine_id]
            engine_external_hits = hit_tokens_external_new[pod][engine_id] \
                - hit_tokens_external[pod][engine_id]
            engine_rows.append(build_row(engine_scope, engine_hbm_hits, engine_hbm_queries,
                                         engine_external_hits, engine_external_queries))

        # 合并当前 POD 下所有 engine_id 的指标增量，用于 ALL_PODS 汇总
        pod_hbm_queries = sum(query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
                              for engine_id in engine_ids)
        pod_hbm_hits = sum(hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
                           for engine_id in engine_ids)
        pod_external_queries = sum(query_tokens_external_new[pod][engine_id]
                                   - query_tokens_external[pod][engine_id] for engine_id in engine_ids)
        pod_external_hits = sum(hit_tokens_external_new[pod][engine_id]
                                - hit_tokens_external[pod][engine_id] for engine_id in engine_ids)

        # 每个 POD 按 engine 升序输出，POD 之间用分隔线隔开
        pod_sections.append(engine_rows)
        total_hbm_queries += pod_hbm_queries
        total_hbm_hits += pod_hbm_hits
        total_external_queries += pod_external_queries
        total_external_hits += pod_external_hits

    if not pod_sections:
        logging.warning(f"No valid prefix cache metrics found. Skipped PODs: {', '.join(skipped_pods)}")
        return

    all_pods_row = build_row("ALL_PODS", total_hbm_hits, total_hbm_queries,
                             total_external_hits, total_external_queries)
    skipped_pods_line = f"Skipped PODs: {', '.join(skipped_pods)}" if skipped_pods else None

    # 控制台只输出所有有效 POD 的汇总
    console_lines = ["=" * line_width, "Prefix cache hit summary", "=" * line_width, header,
                     "-" * line_width, all_pods_row, "=" * line_width]

    # 日志输出每个 POD 下所有 engine 的详细信息，POD 之间用分隔线隔开
    log_lines = ["=" * line_width, "Prefix cache hit detail", "=" * line_width, header,
                 "-" * line_width]
    for idx, section in enumerate(pod_sections):
        if idx > 0:
            log_lines.append("-" * line_width)
        log_lines.extend(section)
    log_lines.extend(["-" * line_width, all_pods_row, "=" * line_width])

    if skipped_pods_line:
        console_lines.append(skipped_pods_line)
        log_lines.append(skipped_pods_line)

    print("\n".join(console_lines))

    try:
        with open("aisbench.log", "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "\n".join(log_lines) + "\n")
    except OSError as error:
        logging.error(f"Failed to write prefix cache information to log files: {error}")

def setup_logging():
    """配置 Python logging：同时输出到控制台、aisbench.log 和 aisbench_all.log"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除已有 handlers（避免重复），但保留会输出到控制台的 StreamHandler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 写入 aisbench.log（本次运行日志）
    file_handler = logging.FileHandler("aisbench.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 写入 aisbench_all.log（所有运行累积日志）
if __name__ == '__main__':
    args = parse_arguments()
    input_len = args.input_len
    output_len = args.output_len
    data_num = args.data_num
    concurrency = args.concurrency
    request_rate = args.request_rate
    test_type = args.test_type
    dataset_path_input = args.dataset
    test_times = args.repeat
    enable_think = args.enable_think
    test_accuracy = args.test_accuracy
    npu_num = args.npu_num
    prefix_num = args.prefix_num
    repeat_rate = parse_prefix_ratio(args.repeat_rate)
    prefix_test = args.prefix_test
    dataset_type = args.dataset_type
    seed = args.seed
    dp = args.dp
    length_mean = args.length_mean
    length_std = args.length_std
    length_min = args.length_min
    length_max = args.length_max

    # 初始化日志文件：清空 aisbench.log（新运行覆盖），aisbench_all.log 添加分隔符（续写）
    with open("aisbench.log", "w", encoding="utf-8") as f:
        f.write("")
    setup_logging()
    logging.info("aisbench.log has been cleared for this run.")

    # 变长参数校验
    if (length_mean is None) ^ (length_std is None):
        raise ValueError("length_mean 和 length_std 必须同时提供或同时不提供")
    if (length_min is None) ^ (length_max is None):
        raise ValueError("length_min 和 length_max 必须同时提供或同时不提供")
    if length_mean is not None and length_mean < 1:
        raise ValueError("length_mean 必须 >= 1")
    if length_std is not None and length_std < 0:
        raise ValueError("length_std 必须 >= 0")
    if length_min is not None and length_min < 1:
        raise ValueError("length_min 必须 >= 1")
    if length_max is not None and length_max < 1:
        raise ValueError("length_max 必须 >= 1")

    logging.info(f"input token length: {input_len}")
    logging.info(f"output token length: {output_len}")
    logging.info(f"number of dataset: {data_num}")
    logging.info(f"concurrency: {concurrency}")
    logging.info(f"request rate: {request_rate}")
    logging.info(f"test type: {test_type}")
    logging.info(f"test_times: {test_times}")
    logging.info(f"v3.1 enable_think: {enable_think}")
    logging.info(f"accuracy test: {test_accuracy}")
    logging.info(f"npu numbers: {npu_num}")
    logging.info(f"prefix numbers: {prefix_num}")
    logging.info(f"dataset repeat rate: {repeat_rate}")
    logging.info(f"test prefix dataset: {prefix_test}")
    logging.info(f"dataset type: {dataset_type}")
    logging.info(f"seed: {seed}")
    logging.info(f"dp size: {dp}")
    logging.info(f"length_mean: {length_mean}")
    logging.info(f"length_std: {length_std}")
    logging.info(f"length_min: {length_min}")
    logging.info(f"length_max: {length_max}")

    # 区分流式和非流式
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = "vllm-api-general-chat"
    elif test_type == "stream":
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"

    if dataset_path_input == "none":
        from dataset_cache import cached
        parameters = [dataset_type, input_len, data_num, prefix_num, repeat_rate, seed, dp,
                      length_mean, length_std, length_min, length_max]
        def generate(destination):
            return create_gsm8k_dataset(dataset_type, input_len, data_num, MODEL_PATH, destination,
                                       prefix_num, repeat_rate, seed, length_mean, length_std, length_min, length_max)
        src_file_prefix, src_file_data = cached(os.environ.get('BENCH_DATASET_CACHE', DATASET_PATH), parameters, MODEL_PATH, generate)
    else:
        # 指定数据集路径逻辑
        if not os.path.exists(dataset_path_input):
            logging.error(f"Dataset {dataset_path_input} is not exist.")
            sys.exit(1)
        src_file_data = dataset_path_input
        src_file_prefix = ""

    dst_dir = os.path.join(WORK_PATH, "ais_bench/datasets/gsm8k")

    # 判断 aisbench 的 gsm8k 文件夹是否存在在
    if not os.path.exists(dst_dir):
        logging.info("dataset work path not exist. creating.")
        os.makedirs(dst_dir)
        logging.info("dataset work path created.")
    # 判断 aisbench 的 gsm8k 文件夹是否存在在 train.jsonl 文件
    train_dataset = os.path.join(dst_dir, "train.jsonl")
    if not os.path.exists(train_dataset):
        logging.info("train dataset not exist. creating.")
        file = open(train_dataset, 'w')
        file.close()
        logging.info("train dataset created.")

    # 生成 aisbench 命令
    ais_bench_cmd = generate_aisbench_command(DEFAULT_PERFORMANCE_TEST)
    logging.info(f"test start, use command: {ais_bench_cmd}")

    # 执行命令
    if dataset_type == "prefix_cache":
        # 前缀数据集测试
        if prefix_test:
            if not POD_INFO:
                pod_info = [HOST_IP+":"+HOST_PORT]
            else:
                pod_info = POD_INFO
            logging.info(f"pod_info: {pod_info}")

            logging.info(f"[开始] 前缀数据集测试"); _benchmark_event('warmup', 'doing')
            # 前缀测试: request rate 固定为 0
            warmup_concurrency = prefix_num if dp == 1 else dp
            modify_aisbench_api(str(warmup_concurrency), "1", request_rate_param="0")
            dst_file = generate_test_dataset(src_file_prefix, dst_dir)

            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

            # Cache filling has only DP*prefix_num requests, so it cannot
            # establish a replenished stable interval. Summarize it separately.
            _benchmark_run(["bash", "-o", "pipefail", "-c", generate_aisbench_command("default_perf")], check=True)
            logging.info(f"[完成] 前缀数据集测试完成，结果保存在aisbench_result.csv"); _benchmark_event('warmup', 'done')

            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)

            # 保存前缀测试结果
            save_result(0, npu_num, repeat_rate)
            logging.info(f"[开始] 全量数据集测试"); _benchmark_event('benchmark', 'doing')
            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            # 执行测试命令
            _benchmark_run(["bash", "-o", "pipefail", "-c", ais_bench_cmd], check=True)
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv"); _benchmark_event('benchmark', 'done')

            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)

        else:
            logging.info(f"[开始] 全量数据集测试"); _benchmark_event('benchmark', 'doing')
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            _benchmark_run(["bash", "-o", "pipefail", "-c", ais_bench_cmd], check=True)
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv"); _benchmark_event('benchmark', 'done')

    else:
        logging.info(f"[开始] 全量数据集测试"); _benchmark_event('benchmark', 'doing')
        modify_aisbench_api(concurrency,str(output_len))
        dst_file = generate_test_dataset(src_file_data, dst_dir)
        if test_times > 1:
            for test_time in range(test_times):
                logging.info(f"Execution rounds: {test_time + 1}")
                _benchmark_run(["bash", "-o", "pipefail", "-c", ais_bench_cmd], check=True)
        else:
            _benchmark_run(["bash", "-o", "pipefail", "-c", ais_bench_cmd], check=True)
        logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv"); _benchmark_event('benchmark', 'done')


    # 保存结果
    save_result(request_rate, npu_num, repeat_rate)
