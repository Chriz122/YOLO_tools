import os
from collections import Counter

base_dir = r"run\1"
classes_txt = r'data\dataset\classes.txt'
splits = ["train", "valid", "test"]

# 讀取類別名稱
if os.path.exists(classes_txt):
    with open(classes_txt, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
else:
    class_names = []

for split in splits:
    label_dir = os.path.join(base_dir, split, "labels")
    class_counter = Counter()
    if not os.path.exists(label_dir):
        print(f"{label_dir} 不存在，略過。")
        continue
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(label_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        class_id = line.strip().split()[0]
                        class_counter[class_id] += 1
    print(f"{split} 資料集各類別數量：")
    for class_id, count in sorted(class_counter.items(), key=lambda x: int(x[0])):
        name = class_names[int(class_id)] if class_names and int(class_id) < len(class_names) else ""
        print(f"    {name}: {count}")
    print("-" * 30)
