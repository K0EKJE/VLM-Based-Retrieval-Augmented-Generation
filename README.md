# VLM-Based-Retrieval-Augmented-Generation

Stanford NLP Project Repo

## Project Structure Tree:
```
VLM RAG/
│
├── benchmark_run_metrics/        # ranking metrics for benchmark
│   ├── datasetName/
│       └── metrics.json           
│
├── codes/
│   ├── finetune.py               # script for fine-tuning retriever using contrastive learning
│   ├── run_benchmark.py          # script to run model on benchmark
│   └── utils                     # util functions
|
├── main/                         # main rag pipeline
│   ├── dbManager.py              # script for article vectorization
│   └── gen.py                    # script for inference and synthetic question generation
│   └── preprocessor.py           # script for doc preprocessing
│
├── requirements.txt              # Python dependencies
```
