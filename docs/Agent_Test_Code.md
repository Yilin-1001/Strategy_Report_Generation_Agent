# Agent 消融实验：质量生成检验与幻觉率检验代码

> 本文档摘录消融实验中两大评估管线的核心代码：
> 1. **质量生成检验** — 5维度评分标准、多轮重复评估、截尾均值聚合、统计分析
> 2. **幻觉率检验** — FActScore方法论的原子事实分解、A/B型验证、三层统计推断

---

## 第一部分：质量生成检验（Quality Evaluation）

### 1.1 评估配置（`config.py`）

```python
# === Evaluation LLM (Qwen via SiliconFlow) ===
EVAL_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
EVAL_BASE_URL = "https://api.siliconflow.cn/v1"
EVAL_MODEL = "Qwen/Qwen3.5-122B-A10B"
EVAL_TIMEOUT = 300

# === Repeated Evaluation Settings ===
NUM_EVAL_ROUNDS = 10       # 每份报告独立打分10次
EVAL_TEMPERATURE = 0.3     # 降低评估温度，受控变异
EVAL_INTERVAL = 8          # 两次打分间隔(秒)
OUTLIER_METHOD = "trimmed_mean"
TRIM_COUNT = 2             # 截尾均值：去掉最高最低各2个，保留中间6个

NUM_RUNS = 3               # 每组实验运行3次
```

### 1.2 5维度评分标准 Prompt（`evaluation/evaluator.py`）

```python
SCORING_SYSTEM_PROMPT = """你是一位拥有20年经验的国际顶尖战略咨询合伙人，专门评估战略规划报告。

你必须严格按JSON格式输出评分，不要输出任何其他文字、分析或解释。不要使用markdown代码块。直接输出纯JSON。

## 评分维度（每项0-20分，允许0.5分精度）

### 维度一：方法论运用与分析框架严谨度 (0-20)
评估各章节是否正确、深入地运用了指定的战略分析模型（PEST/SWOT/BCG/五力/BSC/安索夫/7S/ESG等）。
- 18-20: 模型框架运用精准、维度完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆
- 0-9: 未使用指定模型或使用完全错误

### 维度二：战略一致性与外部环境契合度 (0-20)
评估报告是否紧密结合政策导向（交通强国、交通强省、国企改革），对外部环境认知是否准确深入。
- 18-20: 政策引用精准、外部环境认知深刻、战略定位高度契合
- 14-17: 政策引用较准确但深度不足，或战略定位有偏差
- 10-13: 政策引用泛泛，缺乏针对性分析
- 0-9: 脱离政策背景，战略定位模糊

### 维度三：逻辑连贯性与战略闭环思维 (0-20)
评估诊断阶段→战略推演→实施举措之间是否形成完整闭环，章节间逻辑是否连贯。
- 18-20: 诊断-战略-举措完美闭环，问题-对策一一对应，章节间高度连贯
- 14-17: 基本形成闭环，但部分对策缺乏诊断依据，或章节间有脱节
- 10-13: 闭环不完整，诊断与举措脱节
- 0-9: 各章节独立无关联，无闭环思维

### 维度四：创新性与前瞻洞察力 (0-20)
- 18-20: 提出多维度原创性战略洞察，前瞻判断有力，创新举措兼具突破性与可行性
- 14-17: 有一定的原创洞察和前瞻分析，但创新深度不足或部分流于常规
- 10-13: 以文档信息复述为主，缺乏独立思考和前瞻性判断
- 0-9: 完全依赖文档搬运，无任何原创洞察或前瞻分析

### 维度五：隐性约束洞察与组织治理深度 (0-20)
评估报告是否识别了组织内部的隐性约束（利益相关方博弈、变革阻力、文化惯性），治理策略是否深入。
- 18-20: 深刻洞察组织摩擦力，提出具体可行的变革管理策略
- 14-17: 识别了部分组织约束，但应对策略偏宏观
- 10-13: 对组织约束的认知停留在表面
- 0-9: 忽视组织内部约束，治理建议空泛"""
```

### 1.3 评估 Prompt 构建与章节内容提取

```python
def _extract_chapter_content(report_text: str) -> str:
    """从报告中提取纯章节内容，去除封面/目录/摘要/附录等结构化元素。"""
    lines = report_text.split('\n')
    chapter_lines = []
    in_chapter = False
    in_appendix = False

    for line in lines:
        stripped = line.strip()
        if not in_chapter:
            if re.match(r'^#\s*第[一二三四五六七八九十\d]+章', stripped):
                in_chapter = True
                chapter_lines.append(line)
            continue
        if stripped.startswith('#') and ('附录' in stripped or '战略蓝图' in stripped):
            in_appendix = True
            continue
        if '一致性审查' in stripped:
            break
        if in_appendix:
            continue
        if stripped == '---':
            continue
        chapter_lines.append(line)

    content = '\n'.join(chapter_lines).strip()
    if len(content) < len(report_text) * 0.3:
        return report_text
    return content


def _build_scoring_prompt(report_text: str) -> str:
    """构建评估 Prompt，截断以适配上下文窗口。"""
    chapter_content = _extract_chapter_content(report_text)
    if len(chapter_content) > 25000:
        eval_text = chapter_content[:18000] + "\n\n...[中间章节内容省略]...\n" + chapter_content[-7000:]
    else:
        eval_text = chapter_content

    return f"""请评估以下战略规划报告，直接输出JSON，不要输出其他任何文字。

报告内容:
{eval_text}

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"简评","d2_score":0,"d2_analysis":"简评",
  "d3_score":0,"d3_analysis":"简评","d4_score":0,"d4_analysis":"简评",
  "d5_score":0,"d5_analysis":"简评","total_score":0,"suggestions":"建议"}}

说明: d1=方法论, d2=战略一致, d3=逻辑闭环, d4=创新前瞻, d5=组织治理。
每项0-20分，total=五项之和，analysis和suggestions各限50字以内。"""
```

### 1.4 ReportEvaluator 评估器类

```python
class ReportEvaluator:
    """使用 Qwen 模型进行5维度评分，支持多轮重复评估。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL,
            timeout=EVAL_TIMEOUT
        )
        self.model = EVAL_MODEL

    def evaluate(self, report_text: str) -> Dict:
        """评估单份报告（含5次重试机制）。"""
        prompt = _build_scoring_prompt(report_text)

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=EVAL_TEMPERATURE,
                    max_tokens=8192,
                    extra_body={"enable_thinking": False},
                )
                result_text = response.choices[0].message.content

                if response.choices[0].finish_reason == "length":
                    shorter = _extract_chapter_content(report_text)
                    prompt = _build_scoring_prompt(shorter[:8000])
                    continue

                parsed = self._parse_evaluation(result_text)
                if parsed.get("total_score", 0) > 0:
                    return parsed

            except Exception as e:
                wait = 2 ** attempt
                if "429" in str(e):
                    wait = max(wait, 15)
                if attempt < 4:
                    time.sleep(wait)

        return self._emergency_evaluate(report_text)
```

### 1.5 多轮重复评估与截尾均值聚合

```python
def evaluate_with_repetition(self, report_text: str, num_rounds: int = None) -> Dict:
    """对同一份报告独立打分 num_rounds 次，返回截尾均值聚合结果。"""
    if num_rounds is None:
        num_rounds = NUM_EVAL_ROUNDS  # 默认10轮

    raw_scores = []
    for round_id in range(num_rounds):
        result = self.evaluate(report_text)
        total = result.get("total_score", 0)
        if total > 0:
            raw_scores.append(result)
        if round_id < num_rounds - 1:
            time.sleep(EVAL_INTERVAL)

    if not raw_scores:
        return self._get_fallback_evaluation("所有评估轮次均失败")
    return self._aggregate_scores(raw_scores)


def _aggregate_scores(self, raw_scores: List[Dict]) -> Dict:
    """聚合多次打分结果：截尾均值 + 标准差 + 95%置信区间 + 稳定性标记。"""
    import statistics

    dim_keys = [
        "维度一_方法论运用与分析框架严谨度",
        "维度二_战略一致性与外部环境契合度",
        "维度三_逻辑连贯性与战略闭环思维",
        "维度四_创新性与前瞻洞察力",
        "维度五_隐性约束洞察与组织治理深度",
    ]

    n = len(raw_scores)
    trim = min(TRIM_COUNT, max(0, n // 2 - 1))  # 截尾数

    aggregated = {}
    for dim_key in dim_keys:
        values = sorted([
            s.get("scores", {}).get(dim_key, {}).get("score", 0)
            for s in raw_scores
        ])

        # 截尾均值：去掉最高最低各 trim 个
        trimmed = values[trim : n - trim] if trim > 0 else values
        trimmed_n = len(trimmed)

        mean_val = statistics.mean(trimmed)
        std_val = statistics.stdev(trimmed) if trimmed_n > 1 else 0

        # 95% 置信区间（t 分布）
        if trimmed_n > 1 and std_val > 0:
            t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                       5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                       9: 2.262, 10: 2.228}
            df = trimmed_n - 1
            t_val = t_table.get(df, 2.262 if df > 10 else 2.776)
            ci = t_val * (std_val / (trimmed_n ** 0.5))
        else:
            ci = 0

        is_stable = std_val <= 4.0  # 标准差 ≤ 维度满分的20%

        aggregated[dim_key] = {
            "score": round(mean_val, 1),
            "std": round(std_val, 2),
            "ci_95": round(ci, 2),
            "stable": is_stable,
            "raw_values": values,
        }

    total_mean = round(sum(v["score"] for v in aggregated.values()), 1)
    return {
        "scores": aggregated,
        "total_score": total_mean,
        "num_rounds": n,
        "trim_count": trim,
        "outlier_method": OUTLIER_METHOD,
    }
```

### 1.6 结果导出（CSV + 汇总）

```python
def save_results_csv(results: List[Dict], filepath: str = None) -> str:
    """保存评估结果到 CSV，含置信区间和原始分数。"""
    dim_keys = [
        ("维度一_方法论", "维度一_方法论运用与分析框架严谨度"),
        ("维度二_战略一致", "维度二_战略一致性与外部环境契合度"),
        ("维度三_逻辑闭环", "维度三_逻辑连贯性与战略闭环思维"),
        ("维度四_创新前瞻", "维度四_创新性与前瞻洞察力"),
        ("维度五_组织治理", "维度五_隐性约束洞察与组织治理深度"),
    ]

    fieldnames = ["group", "run", "num_rounds", "total_score"]
    for short_key, _ in dim_keys:
        fieldnames.extend([
            f"{short_key}", f"{short_key}_std",
            f"{short_key}_ci95", f"{short_key}_stable", f"{short_key}_raw",
        ])
    fieldnames.append("improvement_suggestions")

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            scores = r.get("scores", {})
            row = {
                "group": r.get("group", ""),
                "run": r.get("run", ""),
                "num_rounds": r.get("num_rounds", ""),
                "total_score": r.get("total_score", 0),
            }
            for short_key, full_key in dim_keys:
                dim_data = scores.get(full_key, {})
                row[f"{short_key}"] = dim_data.get("score", 0)
                row[f"{short_key}_std"] = dim_data.get("std", 0)
                row[f"{short_key}_ci95"] = dim_data.get("ci_95", 0)
                row[f"{short_key}_stable"] = dim_data.get("stable", "")
                row[f"{short_key}_raw"] = str(dim_data.get("raw_values", []))
            row["improvement_suggestions"] = r.get("improvement_suggestions", "")[:500]
            writer.writerow(row)
    return filepath
```

---

## 第二部分：幻觉率检验（Hallucination Evaluation）

### 2.1 评估配置（`hallucination_eval/config.py`）

```python
# === SiliconFlow API ===
EVAL_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
EVAL_BASE_URL = "https://api.siliconflow.cn/v1"

# === 双模型设置 ===
DECOMPOSE_MODEL = "Pro/moonshotai/Kimi-K2.5"   # 原子分解：需要强中文理解
VERIFY_MODEL = "Pro/moonshotai/Kimi-K2.5"       # 事实验证：只需输出1个词

# === 检索配置 ===
RAG_TOP_K = 5                # B型：RAG 语义搜索结果数
CITED_CHUNKS_TOP_K = 5       # A型：被引来源过滤后的块数
MAX_CONTEXT_TOKENS = 2000    # 发送给验证 LLM 的最大上下文长度
KB_CHUNKS_PATH = DATA_DIR / "all_chunks.json"

# === 并发控制 ===
LLM_CALL_INTERVAL = 2        # LLM 调用间隔(秒)
RAG_CALL_INTERVAL = 1        # RAG 搜索间隔(秒)

# === 统计分析 ===
ALPHA = 0.05
GROUPS = ["group0", "group1", "group2", "group3"]
NUM_RUNS = 1
```

### 2.2 报告章节拆分（`chapter_splitter.py`）

```python
@dataclass
class Chapter:
    index: int           # 0-7
    title: str
    phase: str           # "diagnosis" 或 "initiatives"
    content: str

def split_report(markdown_text: str) -> List[Chapter]:
    """将 Markdown 报告拆分为8个章节。"""
    chapter_pattern = r'^#\s*第([一二三四五六七八]|1|2|3|4|5|6|7|8)章'
    lines = markdown_text.split('\n')
    chapter_starts = []

    for i, line in enumerate(lines):
        match = re.match(chapter_pattern, line)
        if match:
            numeral = match.group(1)
            index_map = {
                '一': 0, '二': 1, '三': 2, '四': 3,
                '五': 4, '六': 5, '七': 6, '八': 7,
                '1': 0, '2': 1, '3': 2, '4': 3,
                '5': 4, '6': 5, '7': 6, '8': 7
            }
            index = index_map.get(numeral, -1)
            if index >= 0:
                chapter_starts.append((index, i, line))

    phase_map = {0: "diagnosis", 1: "diagnosis", 2: "diagnosis",
                 3: "initiatives", 4: "initiatives", 5: "initiatives",
                 6: "initiatives", 7: "initiatives"}

    chapters = []
    for idx, (chapter_idx, start_line, title_line) in enumerate(chapter_starts):
        end_line = chapter_starts[idx + 1][1] if idx < len(chapter_starts) - 1 else len(lines)
        content_lines = lines[start_line:end_line]
        while content_lines and content_lines[-1].strip() == "---":
            content_lines.pop()
        chapters.append(Chapter(
            index=chapter_idx, title=title_line.strip(),
            phase=phase_map[chapter_idx], content='\n'.join(content_lines)
        ))
    return chapters
```

### 2.3 引用解析器（`citation_parser.py`）

```python
@dataclass
class Citation:
    raw_text: str        # 如 "中国交通运输2021_merged"
    position: int
    is_valid: bool       # 是否在 KB 中找到匹配源
    source_file: str

class CitationParser:
    """解析 [来源:xxx] 引用并验证真实性。"""

    def __init__(self, kb_chunks_path: Path = None):
        self.kb_chunks_path = kb_chunks_path or KB_CHUNKS_PATH
        self.source_index = self._build_source_index()

    def _build_source_index(self) -> Dict[str, List[dict]]:
        """构建源文件到其分块的索引。"""
        chunks = json.load(open(self.kb_chunks_path, 'r', encoding='utf-8'))
        index = {}
        for chunk in chunks:
            source = chunk.get('metadata', {}).get('source', '')
            if source:
                if source not in index:
                    index[source] = []
                index[source].append(chunk)
        return index

    def verify_citation(self, citation_text: str) -> Tuple[bool, str]:
        """验证引用来源是否在知识库中存在。"""
        citation_clean = citation_text.strip()
        for source_file in self.source_index:
            source_base = source_file.replace('.txt', '').replace('.pdf', '')
            if citation_clean in source_file or citation_clean in source_base:
                return True, source_file
            if source_base in citation_clean:
                return True, source_file
        return False, ""
```

### 2.4 原子事实分解器（`atomic_decomposer.py`）

**数据结构**：

```python
@dataclass
class AtomicFact:
    text: str           # 原子事实文本
    fact_type: str      # "FACT" 或 "ANALYSIS"
    citation: str       # 引用文本（如有）
    source_index: int   # 原始列表中的索引
```

**分解 Prompt（FActScore 对齐的 V2 版本）**：

```python
ATOMIC_DECOMPOSE_PROMPT = """你是一个事实核查专家。请将以下文本分解为独立的原子事实。

【原子事实定义】
原子事实是一个最简短的、包含单一信息的断言。只要涉及以下任何一类信息，就应标注为 [FACT]：
- 具体实体：人名、机构名、文档名、法规名称、项目名称等
- 具体数据：数字、比例、排名、时间等
- 具体事件：某人说某话、某政策出台、某会议召开等
- 具体状态/性质：某事物被描述为某种状态（如"是基础性产业"）
- 因果/逻辑断言：A导致B、A要求B等可验证的因果关系

只有以下情况才标注为 [ANALYSIS]：
- 纯主观评价（"展现了强大能力"、"具有重要意义"）
- 未来预测/建议（"应加大力度"、"需要重点关注"）
- 方法论描述（"本章将运用XX模型"）
- 纯过渡/连接语句

【引用规则】
- 仅当原文中该句明确包含 [来源:xxx] 标注时，才在原子事实中保留该引用
- 不要为没有引用的原子事实添加引用

【示例】
原文："2024年，集团在省厅考核中排名第二，获评突出单位，展现了强大的战略执行能力。"
分解：
1. [FACT] 集团在2024年省厅考核中排名第二
2. [FACT] 集团获评2024年度全省交通运输工作表现突出单位
3. [ANALYSIS] 集团展现了强大的战略执行能力

原文："交通运输作为国民经济的基础性、先导性产业，其发展环境正经历深刻变革。"
分解：
1. [FACT] 交通运输是国民经济的基础性产业
2. [FACT] 交通运输是国民经济的先导性产业
3. [FACT] 交通运输发展环境正经历深刻变革

【要求】
1. 每个原子事实只包含一个可验证的信息点
2. 保留原文中的关键实体和数据，不要概括或改写
3. 不要遗漏任何事实性断言
4. 格式：序号. [FACT/ANALYSIS] 内容 [来源:xxx]（如有引用）

待分解文本：
{text}

原子事实列表："""
```

**分解器类（含分块处理）**：

```python
class AtomicDecomposer:
    def __init__(self):
        self.client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)

    def decompose_chapter(self, chapter_text: str, max_chunk_chars: int = 3000) -> List[AtomicFact]:
        """分解完整章节，自动分块处理超长文本。"""
        if len(chapter_text) <= max_chunk_chars:
            return self.decompose(chapter_text)

        paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
        all_facts = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_chars and current_chunk:
                chunk_facts = self.decompose(current_chunk)
                all_facts.extend(chunk_facts)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            all_facts.extend(self.decompose(current_chunk))
        return all_facts
```

### 2.5 事实验证器（`fact_verifier.py`）

**验证结果数据结构**：

```python
@dataclass
class Verdict:
    fact_text: str
    fact_type: str       # "FACT" 或 "ANALYSIS"
    citation: str
    category: str        # "A"(有引用), "B"(无引用), "C"(分析性)
    result: str          # 最终判定
    detail: str          # 人类可读解释
```

**三种验证 Prompt**：

```python
# A型：引用验证（有引用标记的事实）
CITATION_VERIFY_PROMPT = """你是一个事实核查专家。请判断以下原子事实是否被给定的参考文档所支持。

原子事实：{fact}

参考文档（被引用的原文片段）：
{source_chunks}

判断要求：
1. 如果参考文档中明确包含与原子事实相符的信息，输出：supported
2. 如果参考文档中的信息与原子事实矛盾，输出：contradicted
3. 如果参考文档中没有相关信息，输出：no_info

仅输出一个判断结果，不要解释。"""

# B型：知识库语义搜索验证（无引用的事实）
KB_VERIFY_PROMPT = """你是一个事实核查专家。请判断以下原子事实是否被给定的知识库检索结果所支持。

原子事实：{fact}

知识库检索结果：
{kb_chunks}

判断要求：
1. 如果检索结果明确支持该事实，输出：supported
2. 如果检索结果明确矛盾，输出：contradicted
3. 如果检索结果没有相关信息，输出：no_info

仅输出一个判断结果。"""

# 共识兜底（KB 无结果时）
LLM_COMMONSENSE_PROMPT = """你是一个事实核查专家。请判断以下原子事实的真实性。

原子事实：{fact}

仅输出以下之一：
- credible: 该事实合理可信，符合常识
- uncertain: 无法确定真实性
- implausible: 该事实看起来不合理或可能有误"""
```

**验证路由逻辑**：

```python
class FactVerifier:
    def verify_fact(self, fact: AtomicFact) -> Verdict:
        """
        验证单个原子事实，路由到A型/B型验证。

        路由规则:
        - ANALYSIS → 直接放行，不计入幻觉指标
        - 有引用 → A型验证（引用真实性 → RAG搜索+源过滤 → LLM比对）
        - 无引用 → B型验证（RAG搜索 → LLM比对 → 共识兜底）
        """
        if fact.fact_type == "ANALYSIS":
            return Verdict(fact_text=fact.text, fact_type=fact.fact_type,
                           citation=fact.citation, category="C",
                           result="analytical",
                           detail="Analytical claim, excluded from hallucination metrics")

        if fact.citation:
            return self._verify_a_type(fact)
        else:
            return self._verify_b_type(fact)

    def _verify_a_type(self, fact: AtomicFact) -> Verdict:
        """A型：引用验证管线。"""
        # Step 1: 验证引用真实性
        is_valid, source_file = self.citation_parser.verify_citation(fact.citation)
        if not is_valid:
            return Verdict(..., result="citation_hallucination",
                           detail=f"Cited source not found in KB: {fact.citation}")

        # Step 2: RAG搜索 + 源过滤
        cited_chunks = self._find_cited_chunks(fact.text, fact.citation)
        if not cited_chunks:
            return self._verify_b_type_no_citation(fact, "A(no_match)")

        # Step 3: LLM验证
        prompt = CITATION_VERIFY_PROMPT.format(fact=fact.text, source_chunks=...)
        result = self._llm_call(prompt)
        # 映射: supported → "supported_by_source", contradicted → "contradicted", else → "unsupported"

    def _verify_b_type_no_citation(self, fact: AtomicFact, category: str) -> Verdict:
        """B型：知识库搜索验证管线。"""
        # Step 1: RAG 语义搜索
        kb_results = self._rag_search(fact.text)

        if kb_results and len(kb_results) >= 2:
            # Step 2: LLM 验证
            prompt = KB_VERIFY_PROMPT.format(fact=fact.text, kb_chunks=...)
            result = self._llm_call(prompt)
            # 映射: supported → "supported_by_kb", contradicted → "contradicted"

        # Step 3: LLM 共识兜底
        prompt = LLM_COMMONSENSE_PROMPT.format(fact=fact.text)
        result = self._llm_call(prompt)
        # 映射: credible → "supported_by_parametric", uncertain → "unverifiable", implausible → "unsupported"
```

### 2.6 指标计算（`metrics.py`）

**FActScore 对齐的指标体系**：

```python
@dataclass
class ReportMetrics:
    group: str
    run: int
    total_facts: int = 0           # N: 总 FACT 项数
    total_analytical: int = 0      # 总 ANALYSIS 项数
    supported_by_source: int = 0   # A型：引用验证通过
    supported_by_kb: int = 0       # B型：KB搜索验证通过
    supported_by_parametric: int = 0  # LLM 共识验证通过
    contradicted: int = 0
    citation_hallucination: int = 0
    unsupported: int = 0
    unverifiable: int = 0

    # 计算指标
    factscore: float = 0.0
    strict_factscore: float = 0.0
    hallucination_rate: float = 0.0
    uncertainty_rate: float = 0.0
    citation_error_rate: float = 0.0
    kb_support_rate: float = 0.0
    parametric_rate: float = 0.0


def compute_report_metrics(verdicts: List[Verdict], group: str, run: int) -> ReportMetrics:
    """
    FActScore = S / N
    where S = supported_by_source + supported_by_kb + supported_by_parametric
    and   N = total_facts
    """
    metrics = ReportMetrics(group=group, run=run)

    for v in verdicts:
        if v.fact_type == "ANALYSIS":
            metrics.total_analytical += 1
            continue
        metrics.total_facts += 1

        if v.citation:
            metrics.total_citations += 1

        # 按结果类型计数
        if v.result == "supported_by_source":
            metrics.supported_by_source += 1
        elif v.result == "supported_by_kb":
            metrics.supported_by_kb += 1
        elif v.result == "supported_by_parametric":
            metrics.supported_by_parametric += 1
        elif v.result == "contradicted":
            metrics.contradicted += 1
        elif v.result == "citation_hallucination":
            metrics.citation_hallucination += 1
            metrics.invalid_citations += 1
        elif v.result == "unsupported":
            metrics.unsupported += 1
        elif v.result == "unverifiable":
            metrics.unverifiable += 1

    N = metrics.total_facts
    if N > 0:
        S = metrics.supported_by_source + metrics.supported_by_kb + metrics.supported_by_parametric
        metrics.factscore = S / N                                    # FActScore
        metrics.strict_factscore = (metrics.supported_by_source + metrics.supported_by_kb) / N  # 仅KB验证
        metrics.hallucination_rate = 1.0 - metrics.factscore         # 幻觉率
        metrics.uncertainty_rate = metrics.unverifiable / N
        metrics.kb_support_rate = (metrics.supported_by_source + metrics.supported_by_kb) / N
        metrics.parametric_rate = metrics.supported_by_parametric / N

    if metrics.total_citations > 0:
        metrics.citation_error_rate = metrics.invalid_citations / metrics.total_citations

    return metrics
```

### 2.7 三层统计推断（`statistical_analysis.py`）

```
Tier 1: 裁定级 Chi-square (N~700, 最高统计功效)
   ↓
Tier 2: 章节级 Two-way ANOVA (N=36, Group × Chapter)
   ↓
Tier 3: 运行级 One-way ANOVA + Tukey HSD (N=12)
```

**Tier 1：裁定级 Chi-square 检验**：

```python
def run_chi_square_test(raw_data_by_group: Dict[str, List[dict]]) -> Dict:
    """Pearson chi-square test on verdict-level 2x4 contingency table.

    每个裁定是一个 FACT 类原子声明，分类为 "supported" 或 "hallucinated"。
    测试幻觉事实比例是否在4组间存在显著差异。
    """
    # 构建 2x4 列联表：行=[supported, hallucinated]，列=[group0..3]
    table = np.zeros((2, len(GROUPS)), dtype=int)
    for col_idx, group in enumerate(GROUPS):
        for raw_data in raw_data_by_group.get(group, []):
            for v in raw_data.get("verdicts", []):
                if v.get("fact_type") != "FACT":
                    continue
                result = v.get("result", "")
                if result in ("supported_by_source", "supported_by_kb", "supported_by_parametric"):
                    table[0, col_idx] += 1
                elif result in ("unsupported", "contradicted", "citation_hallucination", "unverifiable"):
                    table[1, col_idx] += 1

    chi2, p_value, dof, expected = sp_stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))  # 效应量

    # 配对 Chi-square (2x2表) + Bonferroni 校正
    pairwise = [...]
```

**Tier 2：章节级 Two-way ANOVA**：

```python
def run_two_way_anova(raw_data_by_group: Dict[str, List[dict]]) -> Dict:
    """Two-way ANOVA: Group(4) × Chapter(3), N=36 observations.

    测试:
    - Group 主效应: 组间是否存在总体差异？
    - Chapter 主效应: 章节间是否存在总体差异？
    - Group × Chapter 交互: 组间差异是否随章节变化？
    """
    # 构建章节级 DataFrame
    rows = []
    for group in GROUPS:
        for raw_data in raw_data_by_group.get(group, []):
            run_id = raw_data.get("run", 0)
            for cm in raw_data.get("chapter_metrics", []):
                rows.append({
                    "group": group, "run": run_id,
                    "chapter": cm["chapter"],
                    "factscore": cm["factscore"],
                })
    df = pd.DataFrame(rows)

    # Two-way ANOVA with interaction
    model = ols('factscore ~ C(group) * C(chapter)', data=df).fit()
    anova_table = anova_lm(model, typ=2)

    # Partial eta-squared for group effect
    eta_squared_partial = ss_group / (ss_group + ss_residual)

    # Post-hoc: Tukey HSD on group marginal means
    tukey = pairwise_tukeyhsd(df["factscore"], df["group"], alpha=ALPHA)
```

**Tier 3：运行级 One-way ANOVA + Cohen's d**：

```python
def run_anova(report_metrics: Dict[str, List[ReportMetrics]]) -> Dict:
    """One-way ANOVA across 4 groups for each metric."""
    metrics_to_test = [
        "factscore", "strict_factscore", "hallucination_rate",
        "citation_error_rate", "kb_support_rate", "parametric_rate",
        "uncertainty_rate"
    ]
    for metric in metrics_to_test:
        groups_data = [
            [getattr(rm, metric) for rm in report_metrics.get(group, [])]
            for group in GROUPS
        ]
        f_stat, p_value = stats.f_oneway(*groups_data)


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d 效应量。"""
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (mean1 - mean2) / pooled_std
    # |d| < 0.2: negligible, < 0.5: small, < 0.8: medium, >= 0.8: large
```

**Spearman 相关分析**：

```python
def run_spearman_correlation(
    report_metrics: Dict[str, List[ReportMetrics]],
    eval_scores: Dict[str, List[float]]
) -> Dict:
    """幻觉指标与质量评分之间的 Spearman 相关分析。"""
    # FActScore vs 评估总分
    r_fact, p_fact = stats.spearmanr(all_factscores, all_eval_totals)
    # Hallucination Rate vs 评估总分
    r_halluc, p_halluc = stats.spearmanr(all_halluc_rates, all_eval_totals)
```

### 2.8 主评估入口（`run_eval.py`）

```python
def run_single_report(group, run, decomposer, verifier, exp_dir, chapter_filter=None):
    """处理单份报告的完整管线：拆分 → 分解 → 验证 → 指标计算。"""
    report_text = load_report(group, run)
    chapters = split_report(report_text)

    all_verdicts = []
    for chapter in chapters:
        # Step 1: 原子事实分解
        facts = decomposer.decompose_chapter(chapter.content)

        # Step 2: 逐一验证
        for fact in facts:
            verdict = verifier.verify_fact(fact)
            all_verdicts.append(verdict)

    # Step 3: 计算报告级指标
    metrics = compute_report_metrics(all_verdicts, group, run)

    # Step 4: 保存原始结果
    save_raw_results({...}, exp_dir, group, run)
    return metrics


def run_full_evaluation():
    """完整幻觉评估管线（所有章节，所有运行）。"""
    citation_parser = CitationParser()
    decomposer = AtomicDecomposer()
    verifier = FactVerifier(citation_parser)

    for group in GROUPS:
        for run in range(1, NUM_RUNS + 1):
            metrics = run_single_report(group, run, decomposer, verifier, exp_dir)

    # 统计分析
    anova_results = run_anova(report_metrics)              # Tier 3
    tukey_results = run_tukey_hsd(report_metrics)           # 配对比较
    chi_square_results = run_chi_square_test(raw_data)      # Tier 1
    two_way_results = run_two_way_anova(raw_data)           # Tier 2
    cohens_d = {g1_vs_g2: compute_cohens_d(...) for ...}    # 效应量
    correlation = run_spearman_correlation(...)              # 相关分析
```

---

## 架构总结

### 质量生成检验流程

```
报告文本 → _extract_chapter_content() → 仅保留章节正文
                                         ↓
                  SCORING_SYSTEM_PROMPT + _build_scoring_prompt()
                                         ↓
                  Qwen3.5-122B 独立打分 ×10轮 (temperature=0.3)
                                         ↓
                  _aggregate_scores(): 截尾均值(去最高最低各2) + 标准差 + 95%CI
                                         ↓
                  save_results_csv() → evaluation_report.csv
                                         ↓
                  print_summary() → 各组5维度均值±标准差
```

### 幻觉率检验流程

```
报告 → split_report() → 8个Chapter
                            ↓ (逐章)
         decompose_chapter() → List[AtomicFact] (FACT/ANALYSIS)
                            ↓ (逐个fact)
         verify_fact() ─→ ANALYSIS → 直接放行 (不计入指标)
                      ├→ 有引用 → A型验证:
                      │    verify_citation() → RAG搜索+源过滤 → LLM比对
                      │    → supported_by_source / contradicted / citation_hallucination
                      └→ 无引用 → B型验证:
                           RAG语义搜索 → LLM比对 → 共识兜底
                           → supported_by_kb / supported_by_parametric / unsupported

         compute_report_metrics() → FActScore / 幻觉率 / 引用错误率 / ...

         三层统计推断:
         Tier 1: Chi-square (N~700裁定, 组间幻觉比例差异)
         Tier 2: Two-way ANOVA (N=36章节, Group×Chapter交互效应)
         Tier 3: One-way ANOVA + Tukey HSD + Cohen's d (N=12运行)
```

### 指标体系对照表

| 指标 | 定义 | 公式 | 含义 |
|------|------|------|------|
| **FActScore** | 事实准确率 | `(X+Y+Z) / N` | 所有被验证通过的事实比例 |
| **Strict FActScore** | 严格事实率 | `(X+Y) / N` | 仅 KB 验证通过（排除 LLM 共识） |
| **Hallucination Rate** | 幻觉率 | `1 - FActScore` | 未被验证通过的事实比例 |
| **Citation Error Rate** | 引用错误率 | `invalid_citations / total_citations` | 引用了不存在来源的比例 |
| **KB Support Rate** | 知识库支撑率 | `(X+Y) / N` | 可被知识库验证的事实比例 |
| **Parametric Rate** | 参数知识依赖率 | `Z / N` | 依赖 LLM 共识验证的比例 |
| **Uncertainty Rate** | 不确定率 | `unverifiable / N` | 无法验证的事实比例 |
