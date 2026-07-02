# Result and Evaluation Summary

Generated from the local repository on the current app code path.

## Current App Retrieval Evaluation

This rerun evaluates the current deterministic retrieval/ranking path over all 500 test-set rows. It loads the FAISS vectorstore, retrieves candidates using the current app retriever, formats the top 5 candidates with current `compute_score`, and compares retrieved candidate IDs with the test-set ground truth.

| Metric | Value |
| --- | ---: |
| Rows evaluated | 500 |
| Top-1 accuracy / Hit@1 | 0.6240 |
| Hit@3 | 0.6660 |
| Hit@5 | 0.6680 |
| Mean Reciprocal Rank | 0.6447 |
| Top-5 misses | 166 |

## Saved RAGAS Experiment Results

These are the already saved model-based RAGAS outputs in `data/main-data/*/evaluation-results.csv`. They were not fully regenerated because the one-row Gemini/RAGAS smoke test timed out on LLM-judged metrics; only answer similarity completed. A full 500-row RAGAS rerun needs a faster/reliable evaluator configuration.

| Experiment | Rows | Context Precision | Context Recall | Faithfulness | Answer Similarity | Parsed Top-1 Accuracy | Bad ID Parses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-3.5 Normal RAG | 500 | 0.7867 | 0.8366 | 0.8869 | 0.6964 | 0.4360 | 0 |
| GPT-3.5 RAG Fusion | 500 | 0.7686 | 0.8367 | 0.8463 | 0.6943 | 0.4260 | 0 |
| GPT-4 RAG Fusion | 500 | 0.7927 | 0.8434 | 0.9450 | 0.7329 | 0.5580 | 0 |

## Interpretation

The current app retrieval path correctly places the ground-truth candidate at rank 1 for 62.40% of the 500 test rows and inside the top 5 for 66.80% of rows. The strongest saved model-based experiment remains GPT-4 + RAG Fusion, with the best context precision, context recall, faithfulness, and answer similarity among the saved CSVs.

Artifacts:
- `submission_assets/current_app_retrieval_evaluation.csv`
- `submission_assets/evaluation_summary.xlsx`
- `submission_assets/evaluation_metrics_comparison.png`
