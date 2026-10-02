# 如何读懂本项目的评估

## 两组题集

- `data/eval_questions.json`：20 个基础问题，关键词与知识库接近，用于基础回归。
- `data/eval_paraphrases.json`：8 个口语改写问题，观察表达变化带来的检索差异。

两组均为人工编写的小型演示集，并非独立盲测；相关文档标注也可能不完整。不要据此声称某模型普遍优于另一模型。

## 指标的定义

设第 i 个问题的相关文档集合为 R_i，返回的前 k 个文档集合为 S_i。

- **Hit Rate@k**：对每个问题判断 R_i 与 S_i 是否有交集，再取平均。
- **Recall@k**：每题的 `交集大小 / 相关文档总数`，再取平均，属于宏平均。
- **MRR@k**：每题首个相关结果的排名倒数，再取平均；前 k 项无相关结果记 0。

例如相关文档是 B、C，检索返回 A、B，则 Hit=1，Recall=1/2，Reciprocal Rank=1/2。测试用例覆盖了这种情况，避免把 Recall 误写成 Hit Rate。

## 公平比较

比较时保持知识库、题集、top-k 相同。BM25 分数与余弦相似度数值不能横向比较，应比较同一题集上的排名指标。评估 JSON 保留每题的相关文档与实际返回文档，可定位失败案例。

基础题集上高分可能只是题目简单、知识库小、术语重合较多。不能把检索 Recall 写成“问答准确率”。回答是否忠实、引用是否支持结论、未知问题能否拒答，需要另行评估。

## 可复现操作

```bash
python -m ai_toolkit evaluate --backend bm25 --output outputs/bm25-eval.json
python -m ai_toolkit evaluate --backend semantic --output outputs/semantic-eval.json
python -m ai_toolkit evaluate --backend bm25 --dataset data/eval_paraphrases.json --output outputs/bm25-paraphrases.json
python -m ai_toolkit evaluate --backend semantic --dataset data/eval_paraphrases.json --output outputs/semantic-paraphrases.json
```

原始交付报告见 [examples](../examples/)。日常运行输出放在被 Git 忽略的 `outputs/` 中，避免误提交私人问题。
