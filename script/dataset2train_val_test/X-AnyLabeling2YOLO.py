import os
import shutil
import random

# 設定來源資料夾與目標資料夾
src_dir = r'data\dataset\images_labels'
classes_txt = r'data\dataset\classes.txt'
dst_dir = r'run'

# 支援的圖片格式
IMAGE_EXTENSIONS = ('.tif', '.bmp', '.png', '.jpg', '.jpeg')

# 設定資料集比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 1 - (train_ratio + val_ratio)

# 由 classes.txt 讀取類別名稱
with open(classes_txt, 'r', encoding='utf-8') as f:
    names = [line.strip() for line in f if line.strip()]

# 取得所有圖片和.txt檔案，並配對
img_files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
txt_files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
img_basenames = set(os.path.splitext(f)[0] for f in img_files)
txt_basenames = set(os.path.splitext(f)[0] for f in txt_files)
paired_basenames = list(img_basenames & txt_basenames)

# 隨機分配到train/val/test
random.shuffle(paired_basenames)
total = len(paired_basenames)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)
train_set = paired_basenames[:train_end]
val_set = paired_basenames[train_end:val_end]
test_set = paired_basenames[val_end:]

splits = {
    'train': train_set,
    'valid': val_set,
    'test': test_set
}

# 建立目標資料夾結構
def create_output_folder():
    base_dir = os.path.join(dst_dir)
    os.makedirs(base_dir, exist_ok=True)
    # 找出現有資料夾名稱為數字的最大值
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    max_num = max([int(d) for d in existing], default=0)
    folder_name = str(max_num + 1)
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path
full_path = create_output_folder()

for split in splits:
    os.makedirs(os.path.join(full_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(full_path, split, 'labels'), exist_ok=True)

# 複製檔案
for split, basenames in splits.items():
    for name in basenames:
        # 找出對應的圖片檔案
        img_file = next(f for f in img_files if os.path.splitext(f)[0] == name)
        img_ext = os.path.splitext(img_file)[1]
        
        src_img = os.path.join(src_dir, img_file)
        src_lbl = os.path.join(src_dir, name + '.txt')
            
        # 使用 full_path 作為基底路徑
        dst_img = os.path.join(full_path, split, 'images', name + img_ext)
        dst_lbl = os.path.join(full_path, split, 'labels', name + '.txt')
        
        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        
        # 複製檔案
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

# 統計每個 split 各類別數量
def count_classes(label_dir, num_classes):
    counts = [0] * num_classes
    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if 0 <= class_id < num_classes:
                            counts[class_id] += 1
    return counts

# 建立一個字串來儲存資訊
data_info = ""
for split in ['train', 'valid', 'test']:
    label_dir = os.path.join(full_path, split, 'labels')
    class_counts = count_classes(label_dir, len(names))
    split_info = f"{split} 資料集各類別數量:\n"
    print(split_info, end='')
    data_info += split_info
    
    for idx, count in enumerate(class_counts):
        class_info = f"  {names[idx]}: {count}\n"
        print(class_info, end='')
        data_info += class_info
    
    separator = '-' * 30 + '\n'
    print(separator, end='')
    data_info += separator

ratio_info = f"train:val:test : {len(train_set)}:{len(val_set)}:{len(test_set)}\n"
print(ratio_info, end='')
data_info += ratio_info

# 寫入 data_info.txt
with open(os.path.join(full_path, 'data_info.txt'), 'w', encoding='utf-8') as f:
    f.write(data_info)

# 產生 data.yaml
yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(names)}
names: {names}
"""

with open(os.path.join(full_path, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write(yaml_content)


print('YOLO 資料集已建立於:', full_path)