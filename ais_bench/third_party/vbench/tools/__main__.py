"""CLI 入口：python -m vbench.tools

实际生成需在 Python 中调用 run_vbench_generation(generate_fn=...) 并传入你的模型推理函数。
"""
import sys


def main():
    print("""VBench 视频生成工具

用法（在 Python 脚本中）:

    from vbench.tools.video_generator import run_vbench_generation

    def my_generate(prompt: str, index: int):
        video = model.generate(prompt, seed=index)
        return video  # numpy (T,H,W,C) uint8 或 已保存路径

    # Custom 模式
    run_vbench_generation(
        generate_fn=my_generate,
        output_dir="./my_videos",
        mode="custom",
        prompt_source=["a cat running", "a dog swimming"],
    )

    # Standard 单维度（自动使用官方维度->prompt suite 映射）
    run_vbench_generation(
        generate_fn=my_generate,
        output_dir="./videos",
        mode="standard",
        dimension="overall_consistency",
    )

    # 全维度（all_dimension.txt）
    run_vbench_generation(
        generate_fn=my_generate,
        output_dir="./videos",
        mode="all_dimension",
    )

    # temporal_flickering 自动 25 视频/prompt
    run_vbench_generation(
        generate_fn=my_generate,
        output_dir="./videos",
        mode="standard",
        dimension="temporal_flickering",
    )

然后运行: ais_bench --mode eval --models vbench_eval --datasets vbench_standard
配置 path=./videos

详见: ais_bench/benchmark/configs/datasets/vbench/README.md
""")
    sys.exit(0)


if __name__ == "__main__":
    main()
