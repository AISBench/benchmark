# 复用模板：把测评框架搬到别的 SKILL / 别的仓

这里有两套**能跑通**的最小样例，覆盖两种完全不同的 SKILL 形态：

| 目录 | SKILL 形态 | 怎么评 | 结果 |
|---|---|---|---|
| `demo-suite/` + `demo-skill/` | 有 CLI、有退出码、有产物 | 跑命令，断言 stdout / stderr / exit code | 2/2 通过 |
| `demo-suite-docs/` + `demo-docs-skill/` | 纯文档，没有任何可执行脚本 | 不执行，只读文件做静态断言 | 3/3 通过 |

```sh
cd .agents/skill-eval
python -m skill_eval --suite template/demo-suite --work .eval-work/demo
python -m skill_eval --suite template/demo-suite-docs --work .eval-work/demo-docs
```

## 一、真正要改的地方（6 处）

1. **`suite.json` 的 `skill` 段**：`repo` / `path` / `ref` / `commit` / `local_dir`。
   `local_dir` 相对 suite 目录解析，不是相对当前工作目录。
2. **`suite.json` 的 `adapter`**：怎么调用你的 SKILL。
   - `workdir`：一般是 `{skill_dir}`；
   - `command`：可用的占位符是 `{python} {skill_dir} {suite_dir} {job_spec} {bindings} {run_dir} {case_dir} {mode}`；
   - 只做静态检查就把 `type` 设成 `static`，`command` 留空即可。
3. **`fixtures/<name>/{job_spec,bindings}.json`**：这是「输入」的载体，**名字是约定俗成的，语义由你定**。
   你的 SKILL 如果吃 CSV / YAML / 目录，就把它们放这里，在 `adapter.command` 里用 `{job_spec}` 指过去。
4. **`cases/*.json` 的 `patch`**：JSON Pointer 的 set / remove / append，只能改 JSON。
   输入不是 JSON 时，就别用 patch——直接多写几个 fixture，或用 `probe` 适配器自己造场景。
5. **`expect` 里的断言**：见下方选型表。
6. **`limits`**：把「这套测评不覆盖什么」写进 `suite.json`，报告会自动带出来。
   不写等于默许别人以为你测了全部。

## 二、已验证 / 未验证

诚实列一下，别让模板看起来比它实际能做到的更强：

**已验证**

- 纯 CLI 型：退出码、stdout/stderr、产物文件、产物内容（`demo-suite`，2 例）。
- 纯文档型：frontmatter、相对链接、关键约定是否还在（`demo-suite-docs`，3 例）。
- 真实业务 SKILL：`suites/ascend-llm-deploy-eval`，27 例 / 7 维度。
- **整体拷出仓外、零代码改动运行**：把 `skill_eval/` 和某个 suite 拷到任意目录，`python -m skill_eval` 直接跑（已实测，路径全部相对 suite 解析）。

**未验证（别默认能用）**

- **LLM 对话型 SKILL**（没有 CLI，产出是自然语言）：框架里**没有**现成的 `llm` 适配器。
  你需要自己写 40–60 行：拼 prompt → 调模型 → 把结果整理成 stdout 上的 JSON → 用例用 `probe_json` 断言。
  装载、打分、报告、自检可以复用，但「怎么调模型」必须自己写。
- **需要实机 / 外部服务的 SKILL**：只能评「脚本、配置、产物」这些离线可见的东西，评不了实际效果（吞吐、时延、真实响应质量）。
- **输入不是 JSON 的 SKILL**：`patch` 机制不适用，fixture 与场景构造要自己写。
- **非确定性输出**（同一输入两次结果不同）：现有断言都是确定性等值/包含判断，跑不稳。

## 三、按 SKILL 形态选断言

| SKILL 形态 | 优先用这些断言 |
|---|---|
| 有 CLI、靠退出码区分成败 | `exit_code`、`stdout_contains`、`stderr_contains`、`stderr_any_of` |
| 产出文件 / 目录 | `file_exists`、`file_absent`、`file_set_equals`、`json_equals`、`json_present` |
| 产出脚本 / 文本配置 | `text_contains`、`text_not_contains`（配 `path: "skill:..."` 可直接读被测目录） |
| 有中间层可注入（如 review 判定） | `probe_json`：跑一个小脚本，把结构化结果打到 stdout |
| 纯文档 | `frontmatter`、`relative_links_exist`、`cli_flags`、`text_contains` |
| 有「清单 + 摘要」类工件 | `manifest_integrity`（**AISBench 专属约定**，默认读 `.runtime/manifest.json` 的 `files`；别的仓要换 `path` / `files_pointer`，或干脆不用） |

## 四、三个容易踩的坑

1. **路径全是相对 suite 目录**，不是相对你 `cd` 的地方。`local_dir` 写 `../demo-skill` 指的是 suite 的上一级。
2. **`fixture` 名默认是 `base`**，没有这个目录会直接报错；静态型 suite 不需要 fixture（框架会跳过输入准备）。
3. **`probe` 适配器要自己把 skill 的脚本目录塞进 `sys.path`**，不要指望框架替你做——它只负责传路径。

## 五、复用不了的场景（说在前面）

- 想「一键生成测评集」：做不到。用例是**对照 SKILL.md / docs 的承诺**一条条写出来的，这部分必须人来判断。
- 想「自动判定写得对不对」：只有可判定的部分（链接、frontmatter、关键字、结构）能自动；「这条规则合不合理」仍然要评审。
- 想「跨 SKILL 直接套用例」：不同 SKILL 的契约不一样，只有**框架和断言库**能复用，用例要重写。
