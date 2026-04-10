# 该文件夹对比了不同检索方法的评估报告

### 评测脚本

`scripts/eval_workout_rag.py` 提供四种评测路径对比：

```
python scripts/eval_workout_rag.py                          # 默认：vector / hybrid / hybrid_rerank
python scripts/eval_workout_rag.py --methods all            # 包含 hybrid_rewrite
python scripts/eval_workout_rag.py --methods hybrid_rewrite  # 单独跑 rewrite 对比
```
