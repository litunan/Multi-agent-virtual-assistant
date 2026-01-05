# 🧪 Wangwang-Agent 测试套件

用于量化评估多Agent系统各项指标，生成简历所需的STAR指标数据。

## 📁 目录结构

```
tests/
├── __init__.py
├── test_supervisor_metrics.py    # 多智能体调度测试
├── test_mcp_integration.py       # MCP工具链集成测试
├── test_rag_quality.py           # RAG解析质量测试
├── generate_resume_metrics.py    # 汇总报告生成器
├── test_data/                    # 测试数据
│   ├── test_questions.json       # 标准问答对
│   └── test_scenarios.json       # 测试场景
└── test_results/                 # 测试结果输出
```

## 🚀 快速开始

### 1. 运行快速测试（推荐）

不调用LLM，快速获取基础指标：

```bash
cd tests
python generate_resume_metrics.py --quick
```

### 2. 运行完整测试

调用LLM进行完整评估（需要较长时间）：

```bash
cd tests
python generate_resume_metrics.py --full
```

### 3. 单独运行某个测试模块

```bash
# 多智能体调度测试
python test_supervisor_metrics.py

# MCP工具链测试
python test_mcp_integration.py

# RAG质量测试
python test_rag_quality.py
```

## 📊 测试指标说明

### 多智能体调度测试 (`test_supervisor_metrics.py`)

| 指标 | 说明 |
|:--|:--|
| Agent路由准确率 | Supervisor能否正确将任务分配给对应Agent |
| 跨Agent任务成功率 | 需要多Agent协作的复杂任务完成率 |
| 支持对话轮数 | 系统能维持的有效对话轮数 |
| 上下文保持率 | 对话中引用历史上下文的能力 |

### MCP工具链测试 (`test_mcp_integration.py`)

| 指标 | 说明 |
|:--|:--|
| 沙盒安全率 | 路径穿透攻击阻止率 |
| 代码精简率 | MCP vs 传统工具的代码量对比 |
| 工具接入时间 | 新工具接入效率提升 |

### RAG质量测试 (`test_rag_quality.py`)

| 指标 | 说明 |
|:--|:--|
| 文档解析完整度 | MinerU解析保留的结构信息比例 |
| 关键词命中率 | 检索结果包含预期关键词的比例 |
| 答案相关性评分 | LLM评估的答案质量 |

## 📝 输出示例

运行测试后，将在 `test_results/` 目录生成：

- `resume_metrics_YYYYMMDD_HHMMSS.json` - JSON格式完整数据
- `resume_metrics_YYYYMMDD_HHMMSS.md` - Markdown格式报告

### 简历描述示例

```
1. 设计基于状态机的Supervisor调度中枢，通过UserID/SessionID实现用户长期记忆管理，
   成功支持10轮以上复杂长对话，任务拆解成功率达到80%

2. 基于MCP协议标准化工具接口，集成高德地图API与本地文件沙盒环境，
   新工具接入时间缩短50%，文件操作100%限制在安全目录内

3. 针对PDF/Markdown格式混乱问题，采用MinerU进行高精度解析与递归切分，
   文档解析完整度提升至85%，大幅提升RAG Agent回答精准度
```

## ⚠️ 注意事项

1. 完整测试需要配置好API密钥（阿里云百炼、高德地图等）
2. 确保向量数据库已构建（`mcp_course_materials_db/`）
3. 测试过程中会调用实际的Agent，请确保相关服务可用

## 🔧 自定义测试

可以编辑 `test_data/` 目录下的JSON文件来自定义测试用例：

- `test_questions.json` - 添加新的测试问题
- `test_scenarios.json` - 添加新的测试场景

---

*Author: Wangwang-Agent Team*
*Date: 2026-01-04*
