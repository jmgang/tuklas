# Tuklas

Code behind **Tuklas**, a 38.6 MB static embedding model for Tagalog retrieval. Built for the PyCon Davao 2026 talk
*Your Embedding Model Might Be Overqualified for the Job: Building Cheaper and Faster Embedding Models with
Model2Vec, TokenLearn and PyTorch*.

- **Model:** [huggingface.co/jmgang/tuklas-tagalog-512d](https://huggingface.co/jmgang/tuklas-tagalog-512d)
- **Teacher:** [aisingapore/SEA-LION-E5-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-E5-Embedding-600M)

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("jmgang/tuklas-tagalog-512d")
vectors = model.encode(["Kailan naitatag ang Komonwelt ng Pilipinas?"])
```

## How it was built

1. **Model2Vec** distils the teacher into a static table: embed every token, PCA to 512 dimensions, then
   Zipf + SIF weighting.
2. **TokenLearn** fits the table to the teacher's vectors for whole Tagalog Wikipedia passages.
3. **Fine-tuning on question–passage pairs**, with in-batch negatives.
4. **Fine-tuning on mined hard negatives**: the student's own top wrong hits, re-mined during training.
5. **int8 export and vocabulary pruning** to 69,793 rows.

Training ran as one SageMaker job on a single GPU (`scripts/aws_multiwikiqa_stage1_sealion.py`).

## Results

MultiWikiQA Tagalog: 3,984 held-out questions, 1,294 articles, indexed as 40,152 chunks of 192 characters.

| model | model files | Recall@10 |
|---|---:|---:|
| SEA-LION-E5-600M (teacher) | 2,257 MB | 98.59 |
| multilingual-e5-small | 488 MB | 95.01 |
| **Tuklas** | **38.6 MB** | **93.57** |
| potion-multilingual-128M | 531 MB | 73.95 |

## Layout

| path | what it holds |
|---|---|
| `src/pycon_model2vec_demo/` | data preparation, training helpers and evaluation (Recall, MRR, nDCG, paired bootstrap) |
| `scripts/` | SageMaker launchers and workers, local evaluation and comparison runs |
| `configs/` | frozen experiment configurations |
| `tests/` | unit tests |
| `demo/ask.py` | ask Tuklas a Tagalog question and see the ranked articles |

Run artifacts, datasets and model weights are not in this repository. The model is on Hugging Face; AWS account
identifiers in the scripts are replaced with `<aws-account-id>`.

## License

MIT. The training text comes from Tagalog Wikipedia (via MultiWikiQA) and is CC BY-SA.
