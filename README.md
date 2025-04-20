How to use

1. simple
```
python main.py --root_dir /path/to/your/dataset --output_file my_dataset_labels.txt
```
2. train vs val
```
python split_dataset.py \
    --root_dir /path/to/your/dataset \
    --train_ratio 0.9 \
    --train_output train_labels.txt \
    --val_output val_labels.txt \
    --seed 0
```
