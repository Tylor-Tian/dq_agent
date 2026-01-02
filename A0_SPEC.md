# A0｜Demo Spec：数据质量 / 异常检测 Agent（公开/合成数据）

> Repo 目标：做一个**可跑、可测、可出报告**的 Data Quality/Anomaly Detection CLI Agent。

* Demo 名称：`dq_agent`
* 运行方式：`python -m dq_agent demo`
* 输入：一份表（CSV/Parquet）+ 规则配置（YAML/JSON）
* 输出：质量报告（JSON/Markdown）+ 异常解释 + 建议修复动作
* 验收：CLI 可跑、`pytest` 通过、生成报告文件

---

## 1. 目标

### 1.1 业务目标（你能展示什么）

* 任何人拿到 repo：

  1. 运行 demo → 自动生成/下载一份公开/合成数据；
  2. 根据配置执行质量检查 + 异常检测；
  3. 输出一份可读报告（`report.md`）和可机读报告（`report.json`）。

### 1.2 工程目标（你能交付什么）

* **单命令**可跑：`python -m dq_agent demo`
* **可扩展**：新增规则/检测器不需要改核心流程（插件式 registry）
* **可观测**：结构化日志 + 运行指标（耗时/行数/错误数）写入报告
* **可测试**：`pytest -q` 全绿；核心模块单元测试 + CLI 集成测试

### 1.3 不做清单（明确减法）

* 不做“自动修复并回写数据”（只给**建议动作**，不修改原文件）
* 不做分布式大数据引擎（Spark/Flink）——先把单机 demo 做到可展示
* 不做复杂 ML 模型训练平台（只做轻量、可解释的检测方法）
* 不做全量数据目录扫描/血缘/元数据平台（只聚焦单表质量）
* 不做 UI（先 CLI + 报告文件即可）

---

## 2. 用户故事（User Stories）

### US-1：数据工程师快速验表

* 作为数据工程师，我有一份 `orders.parquet` 和一份 `rules.yml`，
* 我希望运行一次命令得到：

  * 哪些字段违反约束（空值、范围、唯一性、枚举、正则）
  * 哪些列存在异常（离群、缺失率突变、重复率突变、类别稀有激增）
  * 具体样例行 + 可能原因 + 下一步修复建议

### US-2：平台化接入（后续演进）

* 作为平台 owner，我希望输出 `report.json` 结构稳定，
* 以后可把 JSON 送到 dashboard / 告警系统。

### US-3：回归验证

* 作为开发者，我希望 `pytest` 覆盖核心逻辑，
* 任何改动都能用快照/金标（golden file）验证报告结构。

---

## 3. 输入与输出

### 3.1 输入

1. 数据文件（必选）

* 支持：CSV / Parquet
* 参数：`--data path/to/file.csv|parquet`

2. 规则配置（必选）

* 支持：YAML（推荐）/ JSON
* 参数：`--config path/to/rules.yml`

3. 可选参数

* `--output-dir ./artifacts`（默认）
* `--format md,json`（默认两者都输出）
* `--sample-rows 20`（报告里抽样展示）
* `--fail-on ERROR`（当出现 ERROR 级别问题时返回非 0）

### 3.2 输出

* `artifacts/<run_id>/report.json`
* `artifacts/<run_id>/report.md`
* `artifacts/<run_id>/run.log`（结构化日志，可选）

#### 报告必须包含

* 概览：行数/列数/运行时间/问题数（按严重级别）
* 规则检查结果：每条规则 PASS/FAIL、失败样本、统计
* 异常检测结果：异常类型、影响列、top 样本、解释
* 建议修复动作：可执行清单（SQL/ETL 层改法、回填策略、上游排查点）

---

## 4. CLI 设计

### 4.1 命令

* `python -m dq_agent run --data <path> --config <path> [--output-dir ...]`
* `python -m dq_agent demo`（一键 demo：生成合成数据 + 使用内置 config 跑一遍）

### 4.2 Demo 行为（必须可离线跑）

* 生成合成数据：`artifacts/demo/orders.parquet`（或 csv）
* 内置规则：`dq_agent/resources/demo_rules.yml`
* 注入“可控异常”（用于评估）：

  * `amount` 出现负数/极大值
  * `user_id` 空值激增
  * `order_id` 重复
  * `status` 出现异常枚举
* 运行后输出报告到：`artifacts/<run_id>/...`

---

## 5. 架构与模块划分

### 5.1 总览（模块图）

```
CLI
  └─ Orchestrator（编排）
       ├─ Loader（CSV/Parquet）
       ├─ Contract Validator（schema/type/required）
       ├─ Rule Engine（规则检查）
       ├─ Profiling（统计概览）
       ├─ Anomaly Detectors（异常检测）
       ├─ Explainer（解释/样本/影响评估）
       ├─ Fix Recommender（建议修复动作）
       └─ Report Writer（JSON/MD）

Observability
  ├─ Logger（结构化日志）
  └─ Metrics（耗时/计数写入 report）
```

### 5.2 模块职责（可测试边界清晰）

* `Loader`

  * `load_table(path) -> DataFrame + meta`
  * 统一列名规范化（trim、lower 可配置）

* `Contract Validator`

  * 依据配置校验：必须列、类型、主键、时间列格式等
  * 输出：`ContractIssue[]`（带 severity）

* `Rule Engine`

  * 执行“确定性规则”：not_null / unique / range / regex / allowed_values / row_count_min 等
  * 输出：`RuleResult[]`（含失败比例、样例行）

* `Profiling`

  * 基本画像：null_rate、distinct_count、top_values、numeric_summary

* `Anomaly Detectors`

  * 执行“统计异常”：

    * Numeric outliers（MAD robust z-score / IQR）
    * Missingness spike（相对阈值 / 绝对阈值）
    * Duplicate spike（重复率阈值）
    * Rare category surge（低频类别占比阈值）
  * 输出：`Anomaly[]`（含 score、阈值、样例）

* `Explainer`

  * 把 rule/anomaly 聚合成“问题单”：

    * 发生在哪些列
    * 影响程度（比例/估计影响行数）
    * 最可能原因（启发式：类型错/上游字段缺失/枚举扩展/连接键异常）

* `Fix Recommender`

  * 规则到动作映射：

    * `unique_fail` → 去重策略（按时间保留最新 / 聚合）+ 上游主键生成检查
    * `not_null_fail` → 回填/默认值/上游字段保证
    * `range_fail` → 单位/币种/符号问题排查
    * `allowed_values_fail` → 枚举字典更新/灰度引入新值
  * 输出：`FixAction[]`（可直接执行的 check list）

* `Report Writer`

  * JSON：稳定 schema，便于机器消费
  * MD：面向人可读（标题、表格、样例块、行动清单）

---

## 6. 技术栈（保持克制但工程化）

* Python：3.11+
* 数据读取：`pandas` + `pyarrow`（读 parquet/写 parquet）
* 配置与契约：`pydantic`（Config schema 校验）
* CLI：`typer`（或 click；typer 更快出体验）
* 报告：

  * JSON：标准库 `json`
  * Markdown：`jinja2` 模板（可选；也可手写拼接）
* 日志：`structlog`（或标准 logging + json formatter）
* 测试：`pytest` + `pytest-cov`
* 代码质量：`ruff`（可选，但建议）

> 约束：不要引入重型依赖（Spark/Great Expectations），先把 demo 做“短小精悍”。

---

## 7. 数据契约（Data Contract）

### 7.1 配置文件结构（YAML 示例）

```yaml
version: 1
dataset:
  name: demo_orders
  primary_key: [order_id]
  time_column: created_at
  expected_row_count:
    min: 1000
columns:
  order_id:
    type: string
    required: true
    checks:
      - unique: true
  user_id:
    type: string
    required: true
    checks:
      - not_null: { max_null_rate: 0.01 }
  amount:
    type: float
    required: true
    checks:
      - range: { min: 0, max: 10000 }
    anomalies:
      - outlier_mad: { z: 6.0 }
  status:
    type: string
    required: true
    checks:
      - allowed_values: { values: ["PAID","REFUND","CANCEL","PENDING"] }
    anomalies:
      - rare_category: { min_rate: 0.001, spike_rate: 0.01 }
  created_at:
    type: datetime
    required: true

report:
  sample_rows: 20
  severity_thresholds:
    error: 0.05
    warn: 0.01
```

### 7.2 契约校验规则（最低集合）

* 列存在性（required）
* 类型可解析（string/int/float/bool/datetime）
* 主键列存在且非空
* 行数下限（可选）

---

## 8. 评估指标（Evaluation）

### 8.1 规则检查（确定性）

* 规则覆盖率：配置中 check 数量、覆盖列数
* 失败率：每条规则的 failing_ratio

### 8.2 异常检测（统计性）

> demo 必须可量化：用“注入异常”生成 ground truth。

* Precision / Recall（以注入的异常行作为正例）
* False Positive Rate（未注入但被判异常的比例）
* Top-k 命中率（报告 top 样本中包含真异常的比例）

### 8.3 性能与成本

* 运行耗时（秒）
* 内存占用（可选：粗略估计）
* 处理行数（吞吐）

---

## 9. 可观测性（Observability）

### 9.1 日志（必须）

* JSON 结构化日志字段：

  * `run_id`, `stage`, `event`, `duration_ms`, `rows`, `cols`, `issues_count`
* 关键阶段必须打点：load / contract / rules / anomalies / report

### 9.2 指标（写入 report.json）

* `timing_ms`：各阶段耗时
* `counts`：规则数/失败数/异常数（按 severity）
* `data_profile`：行数/列数/缺失率概览

### 9.3 失败策略（可控）

* `--fail-on ERROR`：若存在 ERROR 级别问题，CLI exit code = 2
* 配置解析失败/文件不存在：exit code = 1

---

## 10. 测试策略（Test Plan）

### 10.1 单元测试（必须）

* `test_loader_csv_parquet`
* `test_contract_validator_missing_column`
* `test_rule_not_null / unique / range / allowed_values`
* `test_anomaly_outlier_mad`
* `test_fix_recommender_mapping`

### 10.2 集成测试（必须）

* `python -m dq_agent demo` 在临时目录跑通
* 断言输出文件存在：`report.json`、`report.md`
* 断言 report schema 关键字段存在（而不是比对全文）

### 10.3 Golden / Snapshot（推荐）

* 对 `report.json` 做“结构快照”（字段 + 类型）
* Markdown 不做严格快照（避免小改动导致测试脆弱）

---

## 11. 里程碑（Milestones）与验收命令

> 原则：每个里程碑都有**可运行产物**，不做“先学习再说”。

### M0｜Repo 脚手架 + CLI 骨架（当天可完成）

* 交付：可安装/可运行模块入口
* 验收：

  * `python -m dq_agent --help`
  * `pytest -q`（可先只有空测试）

### M1｜Loader + Config + Contract（可跑通最小链路）

* 交付：能读 CSV/Parquet，能解析规则配置，能输出 contract issues
* 验收：

  * `python -m dq_agent run --data ... --config ...`

### M2｜Rule Engine（确定性质量检查）

* 交付：not_null / unique / range / allowed_values 至少 4 类
* 验收：

  * `pytest -q` 覆盖这些规则

### M3｜Anomaly Detectors + Explainer（异常 + 解释）

* 交付：至少 2 类异常检测（numeric outlier + missing spike）
* 验收：

  * demo 注入异常后，报告能指出对应列并给样例

### M4｜Report Writer（JSON/MD）+ Observability

* 交付：固定 JSON schema + 可读 Markdown
* 验收：

  * `artifacts/<run_id>/report.json`、`report.md`、日志字段齐全

### M5｜Demo 一键跑通 + CI（最终验收）

* 交付：`python -m dq_agent demo` 一键出报告
* 验收命令（最终）

  * `python -m dq_agent demo`
  * `pytest -q`
  * `ls artifacts/*/report.json`（或 Windows 等价命令）

---

## 12. 目录结构（建议）

```
dq_agent/
  __init__.py
  __main__.py          # python -m dq_agent
  cli.py               # typer commands: run/demo
  config.py            # pydantic models + load_yaml/json
  loader.py            # csv/parquet loader
  contract.py          # contract validation
  rules/
    __init__.py
    base.py            # Check interface + registry
    checks.py          # not_null/unique/range/allowed_values
  anomalies/
    __init__.py
    base.py            # Detector interface + registry
    detectors.py       # mad_outlier/missing_spike/rare_category
  explain/
    explainer.py
    fix_recommender.py
  report/
    schema.py          # Report dataclasses/pydantic
    writer_json.py
    writer_md.py       # markdown template
  resources/
    demo_rules.yml
  demo/
    generate_demo_data.py

tests/
  test_loader.py
  test_contract.py
  test_rules.py
  test_anomalies.py
  test_cli_demo.py

pyproject.toml
README.md
```

---

## 13. Report JSON Schema（稳定契约）

最低字段（必须）：

* `run_id: str`
* `started_at, finished_at: str`
* `input: {data_path, config_path, format}`
* `summary: {rows, cols, duration_ms, issue_counts:{error,warn,info}}`
* `contract_issues: [ {id, severity, message, column?, sample?} ]`
* `rule_results: [ {rule_id, column?, status, failing_ratio, samples:[...] } ]`
* `anomalies: [ {anomaly_id, column, score, threshold, samples:[...], explanation} ]`
* `fix_actions: [ {severity, title, steps:[...], owner_hint, links?} ]`
* `observability: {timing_ms:{...}, logs_path?}`

---

## 14. 交给 Codex 的工单版任务列表（按文件拆分）

> 说明：下面每一条都是“可直接交给 Codex 生成/修改文件”的粒度。

### 14.1 Repo & Packaging

* **`pyproject.toml`**：定义 package `dq_agent`，依赖（pandas, pyarrow, pydantic, typer, pytest, structlog, jinja2 可选）
* **`dq_agent/__main__.py`**：实现 `python -m dq_agent` 入口，调用 CLI app
* **`README.md`**：写清楚安装/运行/验收命令 + demo 截图/示例输出

### 14.2 CLI

* **`dq_agent/cli.py`**：

  * `run` 命令：参数解析、调用 orchestrator、写报告
  * `demo` 命令：生成 demo 数据 + 使用内置 rules 跑一遍

### 14.3 Config & Contract

* **`dq_agent/config.py`**：

  * pydantic 配置模型（Dataset、Column、Checks、Anomalies、Report）
  * `load_config(path)` 支持 yaml/json
* **`dq_agent/contract.py`**：

  * `validate_contract(df, cfg) -> issues[]`

### 14.4 Loader

* **`dq_agent/loader.py`**：

  * `load_table(path) -> df`
  * 支持 csv/parquet；异常处理；返回 rows/cols meta

### 14.5 Rules Engine

* **`dq_agent/rules/base.py`**：Check interface + registry
* **`dq_agent/rules/checks.py`**：实现 not_null/unique/range/allowed_values

### 14.6 Anomaly Detection

* **`dq_agent/anomalies/base.py`**：Detector interface + registry
* **`dq_agent/anomalies/detectors.py`**：实现 outlier_mad / missing_spike / rare_category

### 14.7 Explain & Fix

* **`dq_agent/explain/explainer.py`**：聚合 rule/anomaly → issues with explanation
* **`dq_agent/explain/fix_recommender.py`**：issue_type → fix_actions 映射

### 14.8 Report

* **`dq_agent/report/schema.py`**：Report 数据结构（pydantic/dataclasses）
* **`dq_agent/report/writer_json.py`**：输出稳定 json
* **`dq_agent/report/writer_md.py`**：Markdown 生成（模板 + 表格）

### 14.9 Demo Data & Resources

* **`dq_agent/demo/generate_demo_data.py`**：生成合成 orders 数据 + 注入异常（可控随机种子）
* **`dq_agent/resources/demo_rules.yml`**：demo 规则配置

### 14.10 Tests

* **`tests/test_rules.py`**：规则单测（构造小 df）
* **`tests/test_anomalies.py`**：异常检测单测（注入已知异常）
* **`tests/test_contract.py`**：缺列/类型错误等
* **`tests/test_cli_demo.py`**：集成测试：跑 demo，断言报告文件存在且 schema 关键字段齐全

### 14.11 Orchestrator（可选单独文件）

* **`dq_agent/orchestrator.py`**：把 loader/contract/rules/anomalies/report 串起来，输出统一 report 对象

---

## 15. 验收清单（你对外展示时就照这个跑）

* [ ] `python -m dq_agent demo` 成功
* [ ] `artifacts/<run_id>/report.json` 存在且包含 schema 必填字段
* [ ] `artifacts/<run_id>/report.md` 可读，含异常解释与修复建议
* [ ] `pytest -q` 通过
