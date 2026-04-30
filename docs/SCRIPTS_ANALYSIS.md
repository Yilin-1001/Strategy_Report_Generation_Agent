# RAG项目脚本使用情���分析报告

**生成日期**：2026-03-10
**分析目的**：区分核心脚本、临时测试脚本和归档脚本

---

## 📊 脚本分类统计

| 类别 | 数量 | 说明 |
|------|------|------|
| **核心生产脚本** | 6个 | 主要Pipeline使用 |
| **临时测试脚本** | 16个 | 根目录下测试脚本 |
| **归档脚本** | 40个 | tests/archive/ 目录 |
| **辅助脚本** | 4个 | scripts/ 目录 |
| **评估脚本** | 6个 | rag_eval/ 目录 |
| **总计** | 72个 | - |

---

## ✅ 核心生产脚本（正在使用）

**位置**：项目根目录

| 脚本名称 | 功能 | 最后使用 | 状态 |
|---------|------|---------|------|
| **clear_milvus.py** | 清空Milvus数据库 | 2026-03-03 | ✅ 生产使用 |
| **chunk_all_documents.py** | 批量分块+索引主脚本 | 2026-03-03 | ✅ 生产使用 ⭐ |
| **convert_pdf_to_txt.py** | PDF转TXT转换 | 可选 | ✅ 生产使用 |
| **batch_merge_lines.py** | 行合并优化 | 推荐 | ✅ 生产使用 |
| **demo_search.py** | 交互式检索演示 | 日常测试 | ✅ 生产使用 |
| **test_retrieval.py** | 自动化检索测试 | 评估时 | ��� 生产使用 |

**使用频率**：每次运行Pipeline必用

---

## 🧪 临时测试脚本（根目录）

**位置**：项目根目录

| 脚本名称 | 类型 | 用途 | 建议 |
|---------|------|------|------|
| **question_generator.py** | 工具 | 生成测试问题 | ✅ 保留 |
| **GPU_Test.py** | 测试 | GPU功能测试 | ⚠️ 可归档 |
| **monitor_indexing.py** | 工具 | 监控索引进度 | ✅ 保留 |
| **organize_project.py** | 工具 | 项目整理 | ⚠️ 可归档 |
| **run_evaluation.py** | 评估 | 运行评估脚本 | ✅ 保留 |
| **test_deduplication.py** | 测试 | 去重验证 | ✅ 保留 |

### 可能不需要的测试脚本

| 脚本名称 | 类型 | 说明 | 建议 |
|---------|------|------|------|
| **chunk_all_documents_no_milvus.py** | 测试 | 不使用Milvus的版本 | ⚠️ 可归档 |
| **test_api_setting.py** | 测试 | API设置测试 | ❌ 可删除 |
| **test_api_simple.py** | 测试 | API简单测试 | ��� 可删除 |
| **test_chunking_no_milvus.py** | 测试 | 无Milvus分块测试 | ❌ 可删除 |
| **test_different_paths.py** | 测试 | 路径测试 | ❌ 可删除 |
| **test_glm_paths.py** | 测试 | GLM路径测试 | ❌ 可删除 |
| **test_manual.py** | 测试 | 手动测试 | ❌ 可归档 |
| **test_manual_ascii.py** | 测试 | ASCII手动测试 | ❌ 可归档 |
| **test_pipeline_tags.py** | 测试 | Pipeline标签测试 | ❌ 可归档 |
| **test_single_file_chunking.py** | 测试 | 单文件分块测试 | ❌ 可归档 |
| **test_single_file_tags.py** | 测试 | 单文件标签测试 | ❌ 可归档 |
| **test_smart_build.py** | 测试 | 智能构建测试 | ❌ 可归档 |
| **test_tags.py** | 测试 | 标签测试 | ❌ 可归档 |
| **test_tags_extraction.py** | 测试 | 标签提取测试 | ❌ 可归档 |

**总计**：14个脚本可以归档或删除

---

## 📦 归档脚本（tests/archive/）

**位置**：tests/archive/
**状态**：已归档，不再使用
**保留原因**：历史参考

### 索引相关（10个）

| 脚本名称 | 说明 |
|---------|------|
| index_knowledge_base.py | 早期索引脚本 |
| index_txt_only.py | 仅索引TXT |
| index_txt_docx_only.py | 索引TXT和DOCX |
| index_skip_docx.py | 跳过DOCX索引 |
| index_with_converted_pdf.py | 使用转换的PDF |
| index_complete_skip_images.py | 完整索引跳过图片 |
| clear_and_reindex.py | 清空并重新索引 |
| clear_and_reindex_gpu.py | GPU版本清空重索引 |
| reindex_all.py | 重新索引所有 |
| index_with_converted_pdf.py | 转换PDF后索引 |

### 单文件测试（7个）

| 脚本名称 | 说明 |
|---------|------|
| test_china_transport_2021.py | 中国交通运输2021测试 |
| test_china_transport_complete.py | 完整版测试 |
| test_china_transport_simple.py | 简化版测试 |
| test_china_aviation_2021.py | 中国民航2021测试 |
| test_v2_metadata.py | V2元数据测试 |
| test_v2_metadata_real.py | V2元数据真实测试 |
| test_page_number_chunking.py | 页码分块测试 |

### 集成测试（4个）

| 脚本名称 | 说明 |
|---------|------|
| test_integration.py | 集成测试 |
| test_integration_data_layer.py | 数据层集成测试 |
| test_integration_data_layer_ascii.py | ASCII数据层测试 |
| test_milvus_connection.py | Milvus连接测试 |
| test_pipeline_logic.py | Pipeline逻辑测试 |

### GPU测试（3个）

| 脚本名称 | 说明 |
|---------|------|
| test_gpu_embedding.py | GPU嵌入测试 |
| test_force_gpu.py | 强制GPU测试 |
| test_pypdf_loader.py | PyPDF加载器测试 |

### 表格处理（6个）

| 脚本名称 | 说明 |
|---------|------|
| test_table_processor.py | 表格处理器测试 |
| test_table_module.py | 表格模块测试 |
| test_table_simple.py | 简单表格测试 |
| test_table_original.py | 原始表格测试 |
| test_table_write.py | 表格写入测试 |

### 行合并相关（5个）

| 脚本名称 | 说明 |
|---------|------|
| test_line_merger.py | 行合并测试 |
| run_line_merger_test.py | 运行行合并测试 |
| simple_merge.py | 简单合并 |
| save_merged_simple.py | 保存简单合并 |
| save_merged_china_2021.py | 保存中国2021合并 |

### 其他（5个）

| 脚本名称 | 说明 |
|---------|------|
| run_china_2021_test.py | 运行中国2021测试 |
| run_retrieval_test.py | 运行检索测试 |
| test_skip_pdf.py | 跳过PDF测试 |

**归档脚本总数**：40个
**建议**：保留作为历史参考，不影响当前功能

---

## 🛠️ 辅助脚本（scripts/）

**位置**：scripts/

| 脚本名称 | 功能 | 状态 |
|---------|------|------|
| **enable_gpu_ocr.py** | 启用GPU OCR | ⚠️ 实验性 |
| **clean_converted_pdf.py** | 清理转换的PDF | ⚠️ 实验性 |
| **restore_original_pdf.py** | 恢复原始PDF | ⚠️ 实验性 |

**建议**：这些是实验性功能，可以保留或归档

---

## 📈 评估脚本（rag_eval/）

**位置**：rag_eval/

| 脚本名称 | 功能 | 状态 |
|---------|------|------|
| **retrieval_eval.py** | 检索评估 | ✅ 使用中 |
| **evaluate_testset.py** | 测试集评估 | ✅ 使用中 |
| **evaluate_simple.py** | 简化评估 | ✅ 使用中 |
| **build_testset_smart.py** | 智能测试集构建 | ✅ 使用中 |
| **build_testset_simple.py** | 简单测试集构建 | ✅ 使用中 |
| **ragas_glm_fixed.py** | RAGAS GLM修复版 | ⚠️ 实验性 |

**建议**：保留评估脚本，用于后续评估

---

## 🗂️ 推荐的项目清理方案

### 方案1：保守清理（推荐）

**创建新目录结构**：
```
RAG Project/
├── 📁 core_scripts/              # 核心生产脚本（移动）
│   ├── clear_milvus.py
│   ├── chunk_all_documents.py
│   ├── convert_pdf_to_txt.py
│   ├── batch_merge_lines.py
│   ├── demo_search.py
│   └── test_retrieval.py
│
├── 📁 utils_scripts/             # 工具脚本（移动）
│   ├── question_generator.py
│   ├── monitor_indexing.py
│   ├── test_deduplication.py
│   └── run_evaluation.py
│
├── 📁 tests_root_archive/        # 根目录测试归档（新建）
│   ├── test_api_*.py
│   ├── test_manual*.py
│   ├── test_tags*.py
│   ├── test_pipeline_tags.py
│   ├── test_single_file*.py
│   ├── test_smart_build.py
│   ├── chunk_all_documents_no_milvus.py
│   ├── test_chunking_no_milvus.py
│   ├── test_different_paths.py
│   ├── test_glm_paths.py
│   └── GPU_Test.py
│
├── 📁 scripts_experimental/      # 实验性脚本（移动）
│   ├── enable_gpu_ocr.py
│   ├── clean_converted_pdf.py
│   └── restore_original_pdf.py
│
└── 📁 tests/archive/             # 已有归档（保持）
    └── [40个文件...]
```

### 方案2：激进清理

**直接删除的脚本**（14个）：
- test_api_setting.py
- test_api_simple.py
- test_chunking_no_milvus.py
- test_different_paths.py
- test_glm_paths.py
- test_manual.py
- test_manual_ascii.py
- test_pipeline_tags.py
- test_single_file_chunking.py
- test_single_file_tags.py
- test_smart_build.py
- test_tags.py
- test_tags_extraction.py
- chunk_all_documents_no_milvus.py

**归档的脚本**：
- GPU_Test.py → tests/archive/
- organize_project.py → tests/archive/

---

## 📋 清理执行脚本

### 自动清理脚本（方案1）

创建 `cleanup_project.py`：

```python
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目，重组脚本结构"""

    project_root = Path("E:/02 Final Year Project/RAG Project")

    # 核心脚本
    core_scripts = [
        "clear_milvus.py",
        "chunk_all_documents.py",
        "convert_pdf_to_txt.py",
        "batch_merge_lines.py",
        "demo_search.py",
        "test_retrieval.py"
    ]

    # 工具脚本
    utils_scripts = [
        "question_generator.py",
        "monitor_indexing.py",
        "test_deduplication.py",
        "run_evaluation.py"
    ]

    # 归档到根目录测试归档
    root_archive_scripts = [
        "test_api_setting.py",
        "test_api_simple.py",
        "test_chunking_no_milvus.py",
        "test_different_paths.py",
        "test_glm_paths.py",
        "test_manual.py",
        "test_manual_ascii.py",
        "test_pipeline_tags.py",
        "test_single_file_chunking.py",
        "test_single_file_tags.py",
        "test_smart_build.py",
        "test_tags.py",
        "test_tags_extraction.py",
        "chunk_all_documents_no_milvus.py",
        "GPU_Test.py",
        "organize_project.py"
    ]

    # 实验性脚本
    experimental_scripts = [
        "scripts/enable_gpu_ocr.py",
        "scripts/clean_converted_pdf.py",
        "scripts/restore_original_pdf.py"
    ]

    # 创建目录
    (project_root / "core_scripts").mkdir(exist_ok=True)
    (project_root / "utils_scripts").mkdir(exist_ok=True)
    (project_root / "tests_root_archive").mkdir(exist_ok=True)
    (project_root / "scripts_experimental").mkdir(exist_ok=True)

    # 移动核心脚本
    for script in core_scripts:
        src = project_root / script
        dst = project_root / "core_scripts" / script
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"✅ 移动核心脚本: {script}")

    # 移动工具脚本
    for script in utils_scripts:
        src = project_root / script
        dst = project_root / "utils_scripts" / script
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"✅ 移动工具脚本: {script}")

    # 移动归档脚本
    for script in root_archive_scripts:
        src = project_root / script
        dst = project_root / "tests_root_archive" / script
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"📦 归档脚本: {script}")

    print("\n✨ 清理完成！")
    print(f"核心脚本: {len(core_scripts)} 个")
    print(f"工具脚本: {len(utils_scripts)} 个")
    print(f"归档脚本: {len(root_archive_scripts)} 个")

if __name__ == "__main__":
    cleanup_project()
```

---

## 📊 最终推荐结构

```
RAG Project/
│
├── 📁 rag_project/              # 核心代码包（保持不变）
│   ├── pipeline.py
│   ├── data_loader/
│   ├── embeddings/
│   ├── storage/
│   ├── utils/
│   └── tests/                   # 单元测试（35个，全部通过）
│
├── 📁 core_scripts/             # 核心生产脚本（6个）⭐
│   ├── clear_milvus.py
│   ├── chunk_all_documents.py   # 主入口
│   ├── convert_pdf_to_txt.py
│   ├── batch_merge_lines.py
│   ├── demo_search.py
│   └── test_retrieval.py
│
├── 📁 utils_scripts/            # 工具脚本（4个）
│   ├── question_generator.py
│   ├── monitor_indexing.py
│   ├── test_deduplication.py
│   └── run_evaluation.py
│
├── 📁 rag_eval/                 # 评估模块（保持不变）
│   ├── retrieval_eval.py
│   ├── evaluate_testset.py
│   └── build_testset_smart.py
│
├── 📁 tests_root_archive/       # 根目录测试归档（新建）
│   └── [16个临时测试脚本]
│
├── 📁 tests/archive/            # 历史归档（保持不变）
│   └── [40个归档脚本]
│
├── 📁 scripts/                  # 辅助脚本（保持不变）
│   └── [3个实验性脚本]
│
├── 📁 config/                   # 配置文件
├── 📁 data/                     # 数据目录
├── 📁 docs/                     # 文档
├── 📁 logs/                     # 日志
└── 📁 知识库/                   # 知识库
```

---

## ✅ 清理后的优势

1. **结构清晰**：核心脚本、工具脚本、归档脚本分离
2. **易于维护**：一眼看出哪些是生产脚本
3. **答辩友好**：可以只展示 core_scripts/ 目录
4. **保留历史**：所有脚本都有备份
5. **文档完善**：清楚标记每个脚本的状态

---

## 🎯 答辩建议

**PPT中展示的结构**：
```
核心脚本（6个）
├── clear_milvus.py - 数据库清理
├── chunk_all_documents.py - 批量索引 ⭐主入口
├── convert_pdf_to_txt.py - PDF转换
├── batch_merge_lines.py - 行合并优化
├── demo_search.py - 交互演示
└── test_retrieval.py - 自动测试

工具脚本（4个）
├── question_generator.py
├── monitor_indexing.py
├── test_deduplication.py
└── run_evaluation.py

评估模块（rag_eval/）
├── retrieval_eval.py
├── evaluate_testset.py
└── build_testset_smart.py
```

**不需要提及的**：
- 16个临时测试脚本（已归档）
- 40个历史归档脚本
- 实验性脚本

---

**报告生成时间**：2026-03-10
**建议操作**：执行 cleanup_project.py 进行重组
