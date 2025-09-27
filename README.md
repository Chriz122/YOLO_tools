# YOLO_tools

YOLO 目標檢測工具集

## 項目結構

```
YOLO_tools/
├─ .gitignore                  # Git 忽略配置文件
├─ data/                       # 資料目錄
│  ├─ dataset/                 # 訓練資料集
│  │  ├─ classes.txt          # 類別定義文件
│  │  └─ images_labels/       # 圖像和標註文件
│  └─ [其他圖像文件]          # 待檢測的圖像
│
├─ models/                     # 模型目錄
│  └─ *.pt                    # PyTorch 模型文件
│
├─ run/                       # 運行結果目錄
│  ├─ [檢測結果目錄]/        # 按圖像命名的結果目錄
│  │  ├─ detection.png       # 檢測結果可視化
│  │  └─ heatmap.png        # 熱力圖可視化
│  └─ 1/                     # 訓練相關數據
│      ├─ data_info.txt      # 資料集信息
│      ├─ data.yaml          # YOLO 配置文件
│      ├─ train/             # 訓練集
│      ├─ valid/             # 驗證集
│      └─ test/              # 測試集
│
├─ script/                    # 腳本目錄
│   ├─ dataset2train_val_test/           # 數據集處理腳本
│   │  ├─ dataset_classes_analysis.py     # 類別分析工具
│   │  └─ X-AnyLabeling2YOLO.py          # 標註格式轉換工具
│   │
│   └─ heatmap_QT/                       # 熱力圖生成工具
│       └─ YOLO_heatmap_QTv2.py          # 熱力圖生成主程式
│
└─ README.md（本檔）            # 專案說明與使用指引（請參閱檔案內容）
```