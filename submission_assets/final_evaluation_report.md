# Final Result and Evaluation Report

Generated after improving the current app retriever on July 2, 2026.

## What Changed

- Increased the FAISS/RRF retrieval pool from 5 to 50 candidates.
- Scores the full retrieved pool before selecting the final top 5 candidates.
- Added normalized token, bigram, skill, and role-phrase scoring as a bonus on top of the existing overlap score.
- Added Ollama CPU fallback with `OLLAMA_NUM_GPU=0` to avoid the local CUDA crash.
- Fixed the response prompt so models do not duplicate percent signs in scores.

## Full 500-Row Retrieval Evaluation

| Run | Rows | Hit@1 | Hit@3 | Hit@5 | MRR | Top-5 Misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before improvement | 500 | 0.6240 | 0.6660 | 0.6680 | 0.6447 | 166 |
| After improvement | 500 | 0.8160 | 0.8960 | 0.9100 | 0.8559 | 45 |
| Improvement delta | 0 | 0.1920 | 0.2300 | 0.2420 | 0.2111 | -121 |

The improved app path raises Hit@1 from 62.4% to 81.6%. Hit@5 improves from 66.8% to 91.0%, reducing top-5 misses from 166 to 45.

## Ollama vs Gemini Generation Sample

This uses the improved retriever and the actual app response prompt on 5 sample rows, one from each test set. A full 500-query LLM batch was intentionally not run because Gemini/RAGAS previously timed out on one-row LLM-judged metrics and each real generation call is relatively slow/costly.

| Provider | Model | Rows | Errors | Retrieval Hit@1 | Generated Hit@1 | Generated ID In Retrieved | Kept Retrieval Top1 | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini | gemini-3.5-flash | 5 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 17.81s |
| ollama | gemma3:4b | 5 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 52.77s |

Ollama `llama3` smoke test after CPU fallback:

| Provider | Model | Generated Best ID | Ground Truth ID | Hit@1 | In Retrieved | Latency | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ollama | llama3 | 946 | 946 | 1 | 1 | 91.03s |  |

## Saved RAGAS Metrics

These are the already saved 500-row RAGAS outputs in the repository. They are included for continuity with the original research results.

| Experiment | Rows | Context Precision | Context Recall | Faithfulness | Answer Similarity | Parsed Top-1 Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-3.5 Normal RAG | 500 | 0.7867 | 0.8366 | 0.8869 | 0.6964 | 0.4360 |
| GPT-3.5 RAG Fusion | 500 | 0.7686 | 0.8367 | 0.8463 | 0.6943 | 0.4260 |
| GPT-4 RAG Fusion | 500 | 0.7927 | 0.8434 | 0.9450 | 0.7329 | 0.5580 |

A fresh full RAGAS rerun was not practical in this environment. The one-row Gemini/RAGAS smoke test timed out on `context_precision`, `context_recall`, and `faithfulness`; only `answer_similarity` completed. For submission purposes, the strongest current evidence is the full deterministic 500-row ranking evaluation above plus the saved RAGAS tables.

## Best Current Configuration

Best retrieval/ranking configuration: improved current app retriever, Hit@1 81.6%, Hit@5 91.0%.

Best live generation provider from the bounded sample: Gemini, because it matched Ollama sample accuracy with lower average latency on this machine. Ollama remains usable locally with `OLLAMA_NUM_GPU=0`, and `gemma3:4b` is the faster local option compared with the `llama3` smoke result.

Artifacts:
- `submission_assets/improved_app_retrieval_evaluation.csv`
- `submission_assets/provider_generation_comparison.csv`
- `submission_assets/provider_generation_summary.csv`
- `submission_assets/ollama_llama3_smoke.csv`
- `submission_assets/final_evaluation_summary.xlsx`
- `submission_assets/final_evaluation_metrics.png`
