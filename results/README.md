# Evaluation results

Run `python evaluate.py --data dataset/data.yaml --weights models/best.pt --split test` to save held-out test-set precision, recall, mAP@0.5, and mAP@0.5:0.95 in `performance_comparison.csv`. The official CRDDC challenge test images do not include annotations, so create a labeled test split from the annotated training data before using this command. Ultralytics also saves plots and detailed evaluation artifacts under `runs/detect/`.

Do not report example or placeholder numbers as measured results. Keep the test set out of model selection, and record the split protocol alongside results.
