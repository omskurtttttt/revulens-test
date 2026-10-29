# RevuLens Pilot Experiment: Performance Comparison Table

| Pipeline | Accuracy | Precision(CG) | Recall(CG) | F1(CG) | Brier | ECE | seconds per review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF + LinearSVC (Val)` | 0.9317 | 0.9338 | 0.9291 | **0.9314** | 0.0525 | 0.0252 | 0.00549 s |
| `TF-IDF + LinearSVC (Test)` | 0.9265 | 0.9249 | 0.9281 | **0.9265** | 0.0527 | 0.0210 | 0.00549 s |
| `distilbert-base-multilingual-cased (Val)` | 0.9465 | 0.9404 | 0.9534 | **0.9469** | 0.0404 | 0.0162 | 0.06066 s |
| `distilbert-base-multilingual-cased (Test - Selected)` | 0.9416 | 0.9442 | 0.9386 | **0.9414** | 0.0439 | 0.0112 | 0.06066 s |
| `xlm-roberta-base (Val)` | 0.9490 | 0.9463 | 0.9519 | **0.9491** | 0.0387 | 0.0134 | 0.12366 s |
| `jcblaise/roberta-tagalog-base (Val)` | 0.9374 | 0.9349 | 0.9400 | **0.9375** | 0.0469 | 0.0162 | 0.13798 s |
