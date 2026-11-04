# dataset/raw_v2/

Drop the VNFairness Label Studio task-export JSON files here (one file per
task, e.g. `task_2_vi.json`), then run:

```
python dataset/convert_labelstudio.py
```

This builds `dataset/processed_v2/{train,val,test}.csv` and
`dataset/metadata/category_mapping_v2.json` for `src/training/train_severity.py`.

See `DATASET_CARD.md` for the dataset's schema and collection methodology.
