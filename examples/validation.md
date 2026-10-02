# 交付验证记录

验证日期：2026-10-03（Asia/Shanghai）。环境：Windows、Python 3.11.4、CPU。核心依赖版本见 [environment.json](environment.json)。

## 已完成的验证

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 自动化测试 | 24 passed | [tests.txt](tests.txt) |
| BM25 中文检索 | 已运行，LoRA 文档排名第一 | [search-demo.txt](search-demo.txt) |
| RAG 请求预览 | 已运行，明确标注无 API 调用 | [rag-dry-run.txt](rag-dry-run.txt) |
| BGE 语义检索 | 已下载真实模型，在 CPU 完成编码与排序 | [semantic-search-demo.txt](semantic-search-demo.txt) |
| 基础与改写题集评估 | 两个后端均已实际运行 | 下表的逐题 JSON |
| OpenAI SDK 请求与响应 | 真实 SDK + 模拟 HTTP 测试通过 | `tests/test_sdk_transport.py` |
| 真实服务生成 | 未验证，未提供服务凭证 | 配置自己的 API 后运行 `ask` |
| GitHub Actions | Windows / Linux × Python 3.10 / 3.12 四种组合全部通过 | [首次发布 CI](https://github.com/tutuhong2001/LLM-Learning-Notes/actions/runs/37035326569) |

发布补充：公开仓库为 [tutuhong2001/LLM-Learning-Notes](https://github.com/tutuhong2001/LLM-Learning-Notes)。上述首次发布 CI 验证的提交为 `e3d850d`；此后补充验证记录的提交仅修改文档。

## 实测结果

| 题集 / 后端 | Hit Rate@3 | Recall@3 | MRR@3 | 原始报告 |
| --- | ---: | ---: | ---: | --- |
| 基础 20 题 / BM25 | 100.00% | 100.00% | 1.0000 | [JSON](bm25-eval.json) |
| 基础 20 题 / BGE | 100.00% | 97.50% | 1.0000 | [JSON](semantic-eval.json) |
| 改写 8 题 / BM25 | 62.50% | 50.00% | 0.4792 | [JSON](bm25-paraphrases.json) |
| 改写 8 题 / BGE | 75.00% | 68.75% | 0.5417 | [JSON](semantic-paraphrases.json) |

这些是小规模人工题集上的检索指标。知识库只有 10 篇短文，基础题关键词重合较多，不能外推为通用效果，也不是问答准确率。

## 值得讨论的失败案例

1. **多篇依据没有全部召回。** “外部知识更新后一定要重新训练模型吗？”的相关标签为 `rag`、`finetuning`。语义检索返回 `rag`、`vector_index`、`inference`，命中了一篇，所以 Hit=1、Recall=0.5。说明只看 Hit Rate 会掩盖证据遗漏。
2. **语义模型也会偏离问题。** “如何让模型学会固定输出格式，同时少训练一些权重？”应命中 `finetuning`，但 BGE 前三项是 `inference`、`rag`、`embeddings`。BM25 在此问题上反而能正确命中。不能预设语义检索必胜。
3. **口语表达仍有困难。** “排第一就找到正确资料与排第三才找到，该怎么量化差别？”两个后端都未在前三项找到 `evaluation`。可以扩充真实表达、改进文档或引入重排，然后在独立测试集复测。

没有依据这些失败案例反复修改标签或题目以抬高分数；保留原始结果用于说明下一步改进方向。相关性标签由项目作者编写，仍可能存在遗漏或歧义。

## 本机依赖问题及处理

本机已有 PyTorch 2.7.0，但原有 torchvision 0.18.1 与 torchaudio 2.3.1 不匹配。复用系统包的项目虚拟环境最初无法导入 Sentence-Transformers；仅在本项目 `.venv` 内安装了匹配的 torchvision 0.22.0、torchaudio 2.7.0 后，实际运行成功。没有升级或删除系统环境中的对应包。

这两个音视频包并非文本检索所需依赖。新用户按照 README 建立独立虚拟环境即可，通常无需安装它们。Windows 下 Hugging Face 的符号链接警告不影响此次模型运行。

## 复现

```bash
python -m pip install -e ".[dev,semantic,llm]"
python scripts/verify.py --semantic
```

在仓库根目录、安装依赖的 Python 环境中运行。首次语义检索需要下载模型。脚本不请求真实生成服务，结果存入 `outputs/`；本目录为交付时真实运行结果的快照。[input-hashes.json](input-hashes.json) 记录知识库与评估集的 SHA-256，便于确认输入是否发生变化。
