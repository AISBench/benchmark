# MME

MME evaluates multimodal models on 14 perception and cognition tasks. Each image has two Yes/No questions, and ACC+ counts an image only when both answers are correct.

The bundled config reads all parquet shards from `C:\需求\MME\MME\data`. It follows InfoVQA's image-first, text-second message construction, uses the parquet `question` verbatim without InfoVQA's extra answer instruction, and sends the embedded image as a base64 data URL with the correct JPEG or PNG MIME type.

```python
from ais_bench.benchmark.configs.datasets.mme.mme_gen_base64 import mme_datasets as datasets
```

Evaluation reports overall and per-task ACC/ACC+, the official Perception and Cognition totals, and the combined MME Score. Official four-column txt files are written below the run result directory in `mme_results/`.
