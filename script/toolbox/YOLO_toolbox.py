import sys
import shutil
import os
import re
import time
import math
import logging
from pathlib import Path
from typing import List, Optional
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog, QCheckBox, QTextEdit, QMessageBox, QSpinBox, QListWidget, QListWidgetItem, QScrollArea, QFrame,
    QSpacerItem, QSizePolicy, QTextBrowser
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtGui import QFont, QPalette, QColor


# 簡單的 logging 設定，方便開發期間除錯
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def list_image_files(directory: Path) -> List[Path]:
    """回傳目錄中的影像檔案 Path 列表。

    會回傳排序過的結果，以維持可重現性。
    """
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        return []
    return [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]

def ensure_dir(path: Path) -> None:
    """確保目錄存在，若不存在則建立之。"""
    path.mkdir(parents=True, exist_ok=True)

def list_weights(directory: Path) -> List[str]:
    """回傳目錄中的 .pt 權重檔案名稱列表。"""
    if not directory or not Path(directory).is_dir():
        return []
    return sorted([p.name for p in Path(directory).iterdir() if p.is_file() and p.suffix == '.pt'])

# -------------------- YOLOv8 自動標註 功能 --------------------
class AnnotateThread(QThread):
    """執行自動標註的執行緒（使用 YOLO 模型）。

    會透過 progress 發出字串訊息，並在完成時發出 finished 訊號。
    """
    progress = Signal(str)
    finished = Signal()

    def __init__(
        self,
        weight: str,
        directory_name: str,
        conf_value: float,
        iou_value: float,
        output_directory: str,
        annotation_type: str,
    ):
        super().__init__()
        self.weight = Path(weight)
        self.directory_name = Path(directory_name)
        self.conf_value = conf_value
        self.iou_value = iou_value
        self.output_directory = Path(output_directory)
        self.annotation_type = annotation_type 

    def run(self) -> None:
        try:
            model = YOLO(str(self.weight))
            self.progress.emit("模型載入成功。\n開始標註...\n")
            start = time.time()

            self.output_directory.mkdir(parents=True, exist_ok=True)

            image_files = list_image_files(self.directory_name)
            total_files = len(image_files)
            self.progress.emit(f"找到 {total_files} 張圖像。\n")

            for filename in image_files:
                self.progress.emit(f"處理 {filename.name}...\n")
                start_fun = time.time()
                color_frame = cv2.imread(str(filename))
                if color_frame is None:
                    self.progress.emit(f"無法讀取 {filename.name}。跳過。\n")
                    continue

                results = model.predict(source=color_frame, verbose=False, device=0, conf=self.conf_value, iou=self.iou_value)

                height, width = color_frame.shape[:2]
                content = self._prepare_annotation_text(results, width, height)

                if not content:
                    self.progress.emit(f"{filename.name} 中未檢測到任何目標。\n")
                    continue

                label_path = self.output_directory / (filename.stem + ".txt")
                label_path.write_text(content, encoding="utf-8")

                end_fun = time.time()
                self.progress.emit(f"完成 {filename.name}，耗時 {end_fun - start_fun:.3f} 秒。\n")

            end = time.time()
            self.progress.emit(f'\n標註完成，總耗時: {end - start:.3f} 秒。\n')
        except Exception as e:
            logging.exception("標註執行緒發生例外")
            self.progress.emit(f"發生錯誤: {e}\n")
        finally:
            self.finished.emit()

    def _prepare_annotation_text(self, results, width: int, height: int) -> str:
        """從模型結果組裝標註文字。

        若沒有偵測到任何框則回傳空字串。
        """
        if len(results[0].boxes) == 0:
            return ""

        segs: List[np.ndarray] = []
        if self.annotation_type == "Instance Segmentation" and getattr(results[0], "masks", None) is not None:
            for seg in results[0].masks.xyn:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segs.append(np.array(seg, dtype=np.int32))

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        lines: List[str] = []
        dw = 1.0 / width
        dh = 1.0 / height

        for idx, (bbox, class_id, score) in enumerate(zip(bboxes, classes, scores)):
            x, y, x2, y2 = bbox
            center_x = (x + x2) / 2.0
            center_y = (y + y2) / 2.0
            w = (x2 - x) * dw
            h = (y2 - y) * dh
            lines.append(f"{class_id} {center_x * dw} {center_y * dh} {w} {h}")

            if self.annotation_type == "Instance Segmentation" and idx < len(segs):
                seg = segs[idx]
                seg_normalized = " ".join([f"{pt[0]/width} {pt[1]/height}" for pt in seg])
                lines.append(seg_normalized)

        return "\n".join(lines) + "\n"

# -------------------- 分配 功能 --------------------
class DistributeThread(QThread):
    progress = Signal(str)
    finished = Signal()

    def __init__(self, image_dir, annotation_dir, num_groups, output_base_dir):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.num_groups = int(num_groups)
        self.output_base_dir = Path(output_base_dir)

    def run(self):
        try:
            self.progress.emit("開始分配圖像...\n")

            image_files = list_image_files(self.image_dir)
            image_names = [p.name for p in image_files]
            total_images = len(image_names)
            self.progress.emit(f"找到 {total_images} 張圖像。\n")

            np.random.shuffle(image_names)

            group_size = math.ceil(total_images / self.num_groups) if self.num_groups > 0 else total_images
            groups = [image_names[i * group_size:(i + 1) * group_size] for i in range(self.num_groups)]

            ensure_dir(self.output_base_dir)

            # 找出已存在的 group_X 資料夾並決定起始編號
            existing_groups = [d.name for d in self.output_base_dir.iterdir() if d.is_dir() and re.match(r'group_\d+', d.name)]
            group_numbers = [int(re.findall(r'group_(\d+)', f)[0]) for f in existing_groups if re.findall(r'group_(\d+)', f)]
            start_num = max(group_numbers) + 1 if group_numbers else 1

            for idx, group in enumerate(groups):
                group_num = start_num + idx
                group_dir = self.output_base_dir / f"group_{group_num}"
                ensure_dir(group_dir)
                self.progress.emit(f"創建分組文件夾: {group_dir}\n")

                for image_file in group:
                    src_image_path = self.image_dir / image_file
                    dst_image_path = group_dir / image_file
                    shutil.copy2(str(src_image_path), str(dst_image_path))

                    if self.annotation_dir:
                        annotation_file = Path(image_file).with_suffix('.txt').name
                        src_annotation_path = self.annotation_dir / annotation_file
                        if src_annotation_path.exists():
                            dst_annotation_path = group_dir / annotation_file
                            shutil.copy2(str(src_annotation_path), str(dst_annotation_path))
                        else:
                            self.progress.emit(f"警告: {annotation_file} 不存在，僅複製圖像。\n")

                self.progress.emit(f"分組 {group_num} 完成，共 {len(group)} 張圖像。\n")

            self.progress.emit("所有分組完成。\n")
        except Exception as e:
            logging.exception("分配執行緒發生例外")
            self.progress.emit(f"發生錯誤: {e}\n")
        finally:
            self.finished.emit()

# -------------------- 主 GUI 應用程式 --------------------
class YOLOTrainerValidatorPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 管理工具")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化 QSettings
        self.settings = QSettings("YourCompany", "YOLOManager")

        # 初始化路徑變數
        self.train_data_path = ""
        self.models_path = ""  # 僅用於訓練
        self.weights_path = ""
        self.val_train_data_path = ""
        self.predict_weights_path = ""
        self.source_path = ""
        self.annotate_image_path = ""
        self.annotate_weights_path = ""
        self.annotate_conf = "0.25"
        self.annotate_iou = "0.45"
        self.annotate_output_directory = ""
        self.annotate_output_display_text = ""

        # 分配模式的變數
        self.distribute_image_path = ""
        self.distribute_annotation_path = ""
        self.distribute_num_groups = 1
        self.distribute_output_base_dir = ""

        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.train_tab = QWidget()
        self.val_tab = QWidget()
        self.predict_tab = QWidget()
        self.annotate_tab = QWidget()  # 標註 Tab
        self.distribute_tab = QWidget()  # 分配 Tab

        self.tabs.addTab(self.train_tab, "訓練")
        self.tabs.addTab(self.val_tab, "驗證")
        self.tabs.addTab(self.predict_tab, "預測")
        self.tabs.addTab(self.annotate_tab, "標註")
        self.tabs.addTab(self.distribute_tab, "分配")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.init_train_tab()
        self.init_val_tab()
        self.init_predict_tab()
        self.init_annotate_tab()  # 初始化標註 Tab
        self.init_distribute_tab()  # 初始化分配 Tab

        # 加載之前的設置
        self.load_settings()

    # -------------------- 分配 分頁 初始化 --------------------
    def init_distribute_tab(self):
        layout = QVBoxLayout()

        # 選擇圖像目錄
        image_dir_layout = QHBoxLayout()
        image_dir_label = QLabel("選擇圖像目錄:")
        self.distribute_image_display = QLineEdit()
        self.distribute_image_display.setReadOnly(True)
        self.distribute_image_button = QPushButton("瀏覽")
        self.distribute_image_button.clicked.connect(self.select_distribute_image_directory)
        image_dir_layout.addWidget(image_dir_label)
        image_dir_layout.addWidget(self.distribute_image_display)
        image_dir_layout.addWidget(self.distribute_image_button)
        layout.addLayout(image_dir_layout)

        # 選擇標註目錄（可選）
        annotation_dir_layout = QHBoxLayout()
        annotation_dir_label = QLabel("選擇標註目錄 (可選):")
        self.distribute_annotation_display = QLineEdit()
        self.distribute_annotation_display.setReadOnly(True)
        self.distribute_annotation_button = QPushButton("瀏覽")
        self.distribute_annotation_button.clicked.connect(self.select_distribute_annotation_directory)
        annotation_dir_layout.addWidget(annotation_dir_label)
        annotation_dir_layout.addWidget(self.distribute_annotation_display)
        annotation_dir_layout.addWidget(self.distribute_annotation_button)
        layout.addLayout(annotation_dir_layout)

        # 選擇分組數量
        num_groups_layout = QHBoxLayout()
        num_groups_label = QLabel("分成幾組:")
        self.distribute_num_groups_spin = QSpinBox()
        self.distribute_num_groups_spin.setMinimum(1)
        self.distribute_num_groups_spin.setMaximum(1000)
        self.distribute_num_groups_spin.setValue(1)
        num_groups_layout.addWidget(num_groups_label)
        num_groups_layout.addWidget(self.distribute_num_groups_spin)
        layout.addLayout(num_groups_layout)

        # 選擇輸出基礎目錄
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("選擇輸出基礎目錄:")
        self.distribute_output_display = QLineEdit()
        self.distribute_output_display.setReadOnly(True)
        self.distribute_output_button = QPushButton("瀏覽")
        self.distribute_output_button.clicked.connect(self.select_distribute_output_directory)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.distribute_output_display)
        output_dir_layout.addWidget(self.distribute_output_button)
        layout.addLayout(output_dir_layout)

        # 開始分配按鈕
        self.distribute_button = QPushButton("開始分配")
        self.distribute_button.clicked.connect(self.start_distribution)
        layout.addWidget(self.distribute_button)

        # 指令輸出與複製按鈕
        distribute_output_layout = QHBoxLayout()
        self.distribute_output = QTextEdit()
        self.distribute_output.setReadOnly(True)
        self.distribute_output.setPlaceholderText("分配進度將顯示在這裡...")
        distribute_output_layout.addWidget(self.distribute_output)

        layout.addLayout(distribute_output_layout)

        self.distribute_tab.setLayout(layout)

    def select_distribute_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇圖像目錄")
        if directory:
            self.distribute_image_path = directory
            self.distribute_image_display.setText(directory)

    def select_distribute_annotation_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇標註目錄 (可選)")
        if directory:
            self.distribute_annotation_path = directory
            self.distribute_annotation_display.setText(directory)

    def select_distribute_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出基礎目錄")
        if directory:
            self.distribute_output_base_dir = directory
            self.distribute_output_display.setText(directory)

    def start_distribution(self):
        if not self.distribute_image_path:
            self.distribute_output.setText("請選擇圖像目錄。")
            return

        if not Path(self.distribute_image_path).exists():
            self.distribute_output.setText("選擇的圖像目錄不存在。")
            return

        num_groups = self.distribute_num_groups_spin.value()
        if num_groups < 1:
            self.distribute_output.setText("分組數量必須至少為1。")
            return

        output_base_dir = Path(self.distribute_output_base_dir) if self.distribute_output_base_dir else None
        if not output_base_dir:
            p = Path(self.distribute_image_path)
            image_dir_name = p.name
            parent_dir = p.parent
            output_base_dir = parent_dir / f"{image_dir_name}_分配結果"
            self.distribute_output_display.setText(str(output_base_dir))

        ensure_dir(output_base_dir)

        annotation_dir = Path(self.distribute_annotation_path) if self.distribute_annotation_path else None

        self.distribute_button.setEnabled(False)
        self.distribute_output.setText("開始分配...\n")

        self.distribute_thread = DistributeThread(
            image_dir=self.distribute_image_path,
            annotation_dir=str(annotation_dir) if annotation_dir else None,
            num_groups=num_groups,
            output_base_dir=str(output_base_dir),
        )
        self.distribute_thread.progress.connect(self.update_distribute_output)
        self.distribute_thread.finished.connect(self.distribution_finished)
        self.distribute_thread.finished.connect(self.distribution_finished)
        self.distribute_thread.start()

    def update_distribute_output(self, message):
        self.distribute_output.append(message)

    def distribution_finished(self):
        self.distribute_output.append("分配完成。\n")
        self.distribute_button.setEnabled(True)

    # -------------------- 標註 分頁 初始化 --------------------
    def init_annotate_tab(self):
        layout = QVBoxLayout()

        # 選擇圖像目錄
        image_dir_layout = QHBoxLayout()
        image_dir_label = QLabel("選擇圖像目錄:")
        self.annotate_image_display = QLineEdit()
        self.annotate_image_display.setReadOnly(True)
        self.annotate_image_button = QPushButton("瀏覽")
        self.annotate_image_button.clicked.connect(self.select_annotate_image_directory)
        image_dir_layout.addWidget(image_dir_label)
        image_dir_layout.addWidget(self.annotate_image_display)
        image_dir_layout.addWidget(self.annotate_image_button)
        layout.addLayout(image_dir_layout)

        # 選擇權重檔案
        weights_layout = QHBoxLayout()
        weights_label = QLabel("選擇權重檔案:")
        self.annotate_weights_combo = QComboBox()
        self.annotate_weights_button = QPushButton("瀏覽權重目錄")
        self.annotate_weights_button.clicked.connect(self.select_annotate_weights_directory)
        weights_layout.addWidget(weights_label)
        weights_layout.addWidget(self.annotate_weights_combo)
        weights_layout.addWidget(self.annotate_weights_button)
        layout.addLayout(weights_layout)

        # 選擇標註類型
        annotation_type_layout = QHBoxLayout()
        annotation_type_label = QLabel("選擇標註類型:")
        self.annotation_type_combo = QComboBox()
        self.annotation_type_combo.addItems(["Object Detection", "Instance Segmentation"])
        annotation_type_layout.addWidget(annotation_type_label)
        annotation_type_layout.addWidget(self.annotation_type_combo)
        layout.addLayout(annotation_type_layout)

        # 輸入參數
        params_layout = QHBoxLayout()

        conf_label = QLabel("Confidence Threshold:")
        self.annotate_conf_input = QLineEdit()
        self.annotate_conf_input.setText(self.annotate_conf)

        iou_label = QLabel("IOU Threshold:")
        self.annotate_iou_input = QLineEdit()
        self.annotate_iou_input.setText(self.annotate_iou)

        params_layout.addWidget(conf_label)
        params_layout.addWidget(self.annotate_conf_input)
        params_layout.addWidget(iou_label)
        params_layout.addWidget(self.annotate_iou_input)

        layout.addLayout(params_layout)

        # 顯示自動生成的輸出目錄
        output_dir_display_layout = QHBoxLayout()
        output_dir_label = QLabel("標註輸出目錄:")
        self.annotate_output_display = QLineEdit()
        self.annotate_output_display.setReadOnly(True)
        output_dir_display_layout.addWidget(output_dir_label)
        output_dir_display_layout.addWidget(self.annotate_output_display)
        layout.addLayout(output_dir_display_layout)

        # 開始標註按鈕
        self.annotate_button = QPushButton("開始標註")
        self.annotate_button.clicked.connect(self.start_annotation)
        layout.addWidget(self.annotate_button)

        # 指令輸出與複製按鈕
        annotate_output_layout = QHBoxLayout()
        self.annotate_output = QTextEdit()
        self.annotate_output.setReadOnly(True)
        self.annotate_output.setPlaceholderText("標註進度將顯示在這裡...")
        annotate_output_layout.addWidget(self.annotate_output)

        layout.addLayout(annotate_output_layout)

        self.annotate_tab.setLayout(layout)

    def select_annotate_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇圖像目錄")
        if directory:
            self.annotate_image_path = str(Path(directory))
            self.annotate_image_display.setText(self.annotate_image_path)
            image_dir_name = Path(self.annotate_image_path).name
            parent_dir = Path(self.annotate_image_path).parent
            self.annotate_output_directory = str(parent_dir / f"{image_dir_name}_auto_annotate_labels")
            self.annotate_output_display.setText(self.annotate_output_directory)

    def select_annotate_weights_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 weights 目錄")
        if directory:
            self.annotate_weights_path = str(Path(directory))
            self.annotate_weights_combo.clear()
            weights_files = list_weights(Path(self.annotate_weights_path))
            if not weights_files:
                self.annotate_weights_combo.addItem("無可用的權重檔案")
            else:
                self.annotate_weights_combo.addItems(weights_files)

    def start_annotation(self):
        if not self.annotate_image_path:
            self.annotate_output.setText("請選擇圖像目錄。")
            return

        if not Path(self.annotate_image_path).exists():
            self.annotate_output.setText("選擇的圖像目錄不存在。")
            return

        if not self.annotate_weights_path:
            self.annotate_output.setText("請選擇 weights 目錄。")
            return

        weights = self.annotate_weights_combo.currentText()
        if weights.startswith("無可用的") or not weights.endswith(".pt"):
            self.annotate_output.setText("請選擇有效的權重檔案。")
            return

        conf = self.annotate_conf_input.text()
        iou = self.annotate_iou_input.text()

        try:
            conf_value = float(conf)
            iou_value = float(iou)
        except ValueError:
            self.annotate_output.setText("請輸入有效的 Confidence 和 IoU 數值。")
            return
        image_dir_name = Path(self.annotate_image_path).name
        parent_dir = Path(self.annotate_image_path).parent
        output_dir = parent_dir / f"{image_dir_name}_auto_annotate_labels"
        ensure_dir(output_dir)

        # 存成字串以供 UI 顯示
        self.annotate_output_directory = str(output_dir)
        self.annotate_output_display.setText(self.annotate_output_directory)

        weights_path = str(Path(self.annotate_weights_path) / weights)
        annotation_type = self.annotation_type_combo.currentText()

        self.annotate_button.setEnabled(False)
        self.annotate_output.setText("開始標註...\n")

        self.annotate_thread = AnnotateThread(
            weight=weights_path,
            directory_name=self.annotate_image_path,
            conf_value=conf_value,
            iou_value=iou_value,
            output_directory=str(output_dir),
            annotation_type=annotation_type,
        )
        self.annotate_thread.progress.connect(self.update_annotate_output)
        self.annotate_thread.finished.connect(self.annotation_finished)
        self.annotate_thread.start()

    def update_annotate_output(self, message):
        self.annotate_output.append(message)

    def annotation_finished(self):
        self.annotate_output.append("標註完成。\n")
        self.annotate_button.setEnabled(True)

    # -------------------- 訓練 分頁 初始化 --------------------
    def init_train_tab(self):
        layout = QVBoxLayout()

        # 選擇 train_data 目錄
        train_data_layout = QHBoxLayout()
        train_data_label = QLabel("選擇 train_data 目錄:")
        self.train_data_display = QLineEdit()
        self.train_data_display.setReadOnly(True)
        self.train_data_button = QPushButton("瀏覽")
        self.train_data_button.clicked.connect(self.select_train_data_directory)
        train_data_layout.addWidget(train_data_label)
        train_data_layout.addWidget(self.train_data_display)
        train_data_layout.addWidget(self.train_data_button)
        layout.addLayout(train_data_layout)

        # 選擇 models 目錄
        models_layout = QHBoxLayout()
        models_label = QLabel("選擇 models 目錄:")
        self.models_display = QLineEdit()
        self.models_display.setReadOnly(True)
        self.models_button = QPushButton("瀏覽")
        self.models_button.clicked.connect(self.select_models_directory)
        models_layout.addWidget(models_label)
        models_layout.addWidget(self.models_display)
        models_layout.addWidget(self.models_button)
        layout.addLayout(models_layout)

        # 選擇數據集
        data_layout = QHBoxLayout()
        data_label = QLabel("選擇數據集:")
        self.train_data_combo = QComboBox()
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.train_data_combo)
        layout.addLayout(data_layout)

        # 選擇模型類型
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("選擇模型類型:")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Detect", "Segment", "Pose", "OBB"])
        self.model_type_combo.currentTextChanged.connect(self.update_model_configs_train)
        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.model_type_combo)
        layout.addLayout(model_type_layout)

        # 選擇模型配置
        model_config_layout = QHBoxLayout()
        model_config_label = QLabel("選擇模型配置:")
        self.model_config_combo = QComboBox()
        model_config_layout.addWidget(model_config_label)
        model_config_layout.addWidget(self.model_config_combo)
        layout.addLayout(model_config_layout)

        # 輸入參數
        params_layout = QHBoxLayout()

        batch_label = QLabel("Batch:")
        self.batch_input = QLineEdit()
        self.batch_input.setText("20")

        epoch_label = QLabel("Epochs:")
        self.epochs_input = QLineEdit()
        self.epochs_input.setText("500")

        imgsz_label = QLabel("Image Size:")
        self.imgsz_input = QLineEdit()
        self.imgsz_input.setText("640")

        params_layout.addWidget(batch_label)
        params_layout.addWidget(self.batch_input)
        params_layout.addWidget(epoch_label)
        params_layout.addWidget(self.epochs_input)
        params_layout.addWidget(imgsz_label)
        params_layout.addWidget(self.imgsz_input)

        layout.addLayout(params_layout)

        # 生成指令按鈕
        self.train_button = QPushButton("生成訓練指令")
        self.train_button.clicked.connect(self.generate_train_command)
        layout.addWidget(self.train_button)

        # 指令輸出與複製按鈕
        train_output_layout = QHBoxLayout()
        self.train_output = QTextEdit()
        self.train_output.setReadOnly(False)
        self.train_output.setPlaceholderText("生成的訓練指令將顯示在這裡，您可以手動修改它...")
        train_output_layout.addWidget(self.train_output)

        self.copy_train_button = QPushButton("複製指令")
        self.copy_train_button.clicked.connect(lambda: self.copy_to_clipboard(self.train_output))
        train_output_layout.addWidget(self.copy_train_button)

        layout.addLayout(train_output_layout)

        self.train_tab.setLayout(layout)

    # -------------------- 驗證 分頁 初始化 --------------------
    def init_val_tab(self):
        layout = QVBoxLayout()

        # 選擇 train_data 目錄
        val_train_data_layout = QHBoxLayout()
        val_train_data_label = QLabel("選擇 train_data 目錄:")
        self.val_train_data_display = QLineEdit()
        self.val_train_data_display.setReadOnly(True)
        self.val_train_data_button = QPushButton("瀏覽")
        self.val_train_data_button.clicked.connect(self.select_val_train_data_directory)
        val_train_data_layout.addWidget(val_train_data_label)
        val_train_data_layout.addWidget(self.val_train_data_display)
        val_train_data_layout.addWidget(self.val_train_data_button)
        layout.addLayout(val_train_data_layout)

        # 選擇 weights 目錄
        weights_layout = QHBoxLayout()
        weights_label = QLabel("選擇 weights 目錄:")
        self.weights_display = QLineEdit()
        self.weights_display.setReadOnly(True)
        self.weights_button = QPushButton("瀏覽")
        self.weights_button.clicked.connect(self.select_weights_directory)
        weights_layout.addWidget(weights_label)
        weights_layout.addWidget(self.weights_display)
        weights_layout.addWidget(self.weights_button)
        layout.addLayout(weights_layout)

        # 選擇數據集
        val_data_layout = QHBoxLayout()
        val_data_label = QLabel("選擇數據集:")
        self.val_data_combo = QComboBox()
        val_data_layout.addWidget(val_data_label)
        val_data_layout.addWidget(self.val_data_combo)
        layout.addLayout(val_data_layout)

        # 選擇模型類型
        val_model_type_layout = QHBoxLayout()
        val_model_type_label = QLabel("選擇模型類型:")
        self.val_model_type_combo = QComboBox()
        self.val_model_type_combo.addItems(["Detect", "Segment", "Pose", "OBB"])
        val_model_type_layout.addWidget(val_model_type_label)
        val_model_type_layout.addWidget(self.val_model_type_combo)
        layout.addLayout(val_model_type_layout)

        # 選擇權重檔案
        selected_weights_layout = QHBoxLayout()
        selected_weights_label = QLabel("選擇權重檔案:")
        self.selected_weights_combo = QComboBox()
        selected_weights_layout.addWidget(selected_weights_label)
        selected_weights_layout.addWidget(self.selected_weights_combo)
        layout.addLayout(selected_weights_layout)

        # 選擇驗證數據集類型
        split_layout = QHBoxLayout()
        split_label = QLabel("選擇驗證數據集類型:")
        self.split_combo = QComboBox()
        self.split_combo.addItems(["valid", "train", "test"])
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.split_combo)
        layout.addLayout(split_layout)

        # 生成指令按鈕
        self.val_button = QPushButton("生成驗證指令")
        self.val_button.clicked.connect(self.generate_val_command)
        layout.addWidget(self.val_button)

        # 指令輸出與複製按鈕
        val_output_layout = QHBoxLayout()
        self.val_output = QTextEdit()
        self.val_output.setReadOnly(False)
        self.val_output.setPlaceholderText("生成的驗證指令將顯示在這裡，您可以手動修改它...")
        val_output_layout.addWidget(self.val_output)

        self.copy_val_button = QPushButton("複製指令")
        self.copy_val_button.clicked.connect(lambda: self.copy_to_clipboard(self.val_output))
        val_output_layout.addWidget(self.copy_val_button)

        layout.addLayout(val_output_layout)

        self.val_tab.setLayout(layout)

    # -------------------- 預測 分頁 初始化 --------------------
    def init_predict_tab(self):
        layout = QVBoxLayout()

        # 選擇 weights 目錄
        predict_weights_layout = QHBoxLayout()
        predict_weights_label = QLabel("選擇 weights 目錄:")
        self.predict_weights_display = QLineEdit()
        self.predict_weights_display.setReadOnly(True)
        self.predict_weights_button = QPushButton("瀏覽")
        self.predict_weights_button.clicked.connect(self.select_predict_weights_directory)
        predict_weights_layout.addWidget(predict_weights_label)
        predict_weights_layout.addWidget(self.predict_weights_display)
        predict_weights_layout.addWidget(self.predict_weights_button)
        layout.addLayout(predict_weights_layout)

        # 選擇圖像來源
        source_layout = QHBoxLayout()
        source_label = QLabel("選擇圖像來源:")
        self.source_display = QLineEdit()
        self.source_display.setReadOnly(True)
        self.source_button = QPushButton("瀏覽")
        self.source_button.clicked.connect(self.select_source_directory)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_display)
        source_layout.addWidget(self.source_button)
        layout.addLayout(source_layout)

        # 選擇模型類型
        predict_model_type_layout = QHBoxLayout()
        predict_model_type_label = QLabel("選擇模型類型:")
        self.predict_model_type_combo = QComboBox()
        self.predict_model_type_combo.addItems(["Detect", "Segment", "Pose", "OBB"])
        predict_model_type_layout.addWidget(predict_model_type_label)
        predict_model_type_layout.addWidget(self.predict_model_type_combo)
        layout.addLayout(predict_model_type_layout)

        # 選擇權重檔案
        predict_selected_weights_layout = QHBoxLayout()
        predict_selected_weights_label = QLabel("選擇權重檔案:")
        self.predict_selected_weights_combo = QComboBox()
        predict_selected_weights_layout.addWidget(predict_selected_weights_label)
        predict_selected_weights_layout.addWidget(self.predict_selected_weights_combo)
        layout.addLayout(predict_selected_weights_layout)

        # 生成指令按鈕
        self.predict_button = QPushButton("生成預測指令")
        self.predict_button.clicked.connect(self.generate_predict_command)
        layout.addWidget(self.predict_button)

        # 指令輸出與複製按鈕
        predict_output_layout = QHBoxLayout()
        self.predict_output = QTextEdit()
        self.predict_output.setReadOnly(False)
        self.predict_output.setPlaceholderText("生成的預測指令將顯示在這裡，您可以手動修改它...")
        predict_output_layout.addWidget(self.predict_output)

        self.copy_predict_button = QPushButton("複製指令")
        self.copy_predict_button.clicked.connect(lambda: self.copy_to_clipboard(self.predict_output))
        predict_output_layout.addWidget(self.copy_predict_button)

        layout.addLayout(predict_output_layout)

        self.predict_tab.setLayout(layout)

    # -------------------- 訓練 分頁 方法 --------------------
    def select_train_data_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 train_data 目錄")
        if directory:
            self.train_data_path = directory
            self.train_data_display.setText(directory)
            self.load_train_datasets()

    def select_models_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 models 目錄")
        if directory:
            self.models_path = directory
            self.models_display.setText(directory)
            self.update_model_configs_train(self.model_type_combo.currentText())

    def load_train_datasets(self):
        self.train_data_combo.clear()
        if not self.train_data_path:
            self.train_data_combo.addItem("未選擇 train_data 目錄")
            return
        p = Path(self.train_data_path)
        if not p.exists() or not p.is_dir():
            self.train_data_combo.addItem("無可用的數據集")
            return
        subfolders = [d.name for d in sorted(p.iterdir()) if d.is_dir()]
        if not subfolders:
            self.train_data_combo.addItem("無可用的數據集")
        else:
            self.train_data_combo.addItems(subfolders)

    def update_model_configs_train(self, model_type):
        self.model_config_combo.clear()
        if not self.models_path:
            self.model_config_combo.addItem("未選擇 models 目錄")
            return
        model_configs_path = Path(self.models_path) / model_type
        if not model_configs_path.exists():
            self.model_config_combo.addItem(f"未找到 {model_type} 目錄")
            return
        yaml_files = [p.name for p in sorted(model_configs_path.iterdir()) if p.is_file() and p.suffix == '.yaml']
        if not yaml_files:
            self.model_config_combo.addItem(f"無可用的 {model_type} 配置檔")
        else:
            self.model_config_combo.addItems(yaml_files)

    def generate_train_command(self):
        dataset = self.train_data_combo.currentText()
        model_type = self.model_type_combo.currentText()
        model_config = self.model_config_combo.currentText()
        batch = self.batch_input.text()
        epochs = self.epochs_input.text()
        imgsz = self.imgsz_input.text()

        if not self.train_data_path:
            self.train_output.setText("請選擇 train_data 目錄。")
            return

        if dataset == "未選擇 train_data 目錄" or dataset == "無可用的數據集":
            self.train_output.setText("請選擇有效的數據集。")
            return

        if not self.models_path:
            self.train_output.setText("請選擇 models 目錄。")
            return

        if model_config.startswith("未找到") or model_config.startswith("無可用的"):
            self.train_output.setText(f"請選擇有效的 {model_type} 模型配置。")
            return

        data_path = Path(self.train_data_path) / dataset / "data.yaml"
        model_config_path = Path(self.models_path) / model_type / model_config
        name = dataset

        if not data_path.exists():
            self.train_output.setText(f"找不到 data.yaml 檔案在 {data_path}。")
            return

        if not model_config_path.exists():
            self.train_output.setText(f"找不到模型配置檔案在 {model_config_path}。")
            return

        try:
            with data_path.open('r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
        except Exception as e:
            self.train_output.setText(f"讀取 data.yaml 失敗: {e}")
            return

        data_yaml['val'] = "../valid/images"

        try:
            with data_path.open('w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.train_output.setText(f"寫入 data.yaml 失敗: {e}")
            return

        command = (
            f"yolo {model_type.lower()} train "
            f"data=\"{data_path}\" "
            f"model=\"{model_config_path}\" "
            f"batch={batch} "
            f"epochs={epochs} "
            f"imgsz={imgsz} "
            f"save_period=100 "
            f"device=0 "
            f"workers=2 "
            f"name={name} "
            f"patience=0"
        )

        self.train_output.setText(command)

    # -------------------- 驗證 分頁 方法 --------------------
    def select_val_train_data_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 train_data 目錄")
        if directory:
            self.val_train_data_path = directory
            self.val_train_data_display.setText(directory)
            self.load_val_datasets()

    def select_weights_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 weights 目錄")
        if directory:
            self.weights_path = directory
            self.weights_display.setText(directory)
            self.load_weights_val()

    def load_val_datasets(self):
        self.val_data_combo.clear()
        # 檢查路徑是否存在並為資料夾，若不存在或未設定則提示並返回
        if not hasattr(self, 'val_train_data_path') or not self.val_train_data_path:
            self.val_data_combo.addItem("未選擇 train_data 目錄")
            return
        p = Path(self.val_train_data_path)
        if not p.exists() or not p.is_dir():
            self.val_data_combo.addItem("未選擇 train_data 目錄")
            return

        subfolders = [d.name for d in sorted(p.iterdir()) if d.is_dir()]
        if not subfolders:
            self.val_data_combo.addItem("無可用的數據集")
        else:
            self.val_data_combo.addItems(subfolders)

    def load_weights_val(self):
        self.selected_weights_combo.clear()
        # 檢查 weights_path 是否存在且為資料夾
        if not self.weights_path or not Path(self.weights_path).is_dir():
            self.selected_weights_combo.addItem("未選擇 weights 目錄")
            return
        weights_files = list_weights(Path(self.weights_path))
        if not weights_files:
            self.selected_weights_combo.addItem("無可用的權重檔案")
        else:
            self.selected_weights_combo.addItems(weights_files)

    def generate_val_command(self):
        dataset = self.val_data_combo.currentText()
        model_type = self.val_model_type_combo.currentText()
        weights = self.selected_weights_combo.currentText()
        split = self.split_combo.currentText()  # valid, train, test

        if not hasattr(self, 'val_train_data_path') or not self.val_train_data_path:
            self.val_output.setText("請選擇 train_data 目錄。")
            return

        if dataset == "未選擇 train_data 目錄" or dataset == "無可用的數據集":
            self.val_output.setText("請選擇有效的數據集。")
            return

        if not self.weights_path:
            self.val_output.setText("請選擇 weights 目錄。")
            return

        if weights.startswith("未選擇") or weights.startswith("無可用的"):
            self.val_output.setText("請選擇有效的權重檔案。")
            return

        data_path_original = Path(self.val_train_data_path) / dataset / "data.yaml"
        if not data_path_original.exists():
            self.val_output.setText(f"找不到 data.yaml 檔案在 {data_path_original}。")
            return

        try:
            with data_path_original.open('r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
        except Exception as e:
            self.val_output.setText(f"讀取 data.yaml 失敗: {e}")
            return

        if split == "train":
            data_yaml['val'] = "../train/images"
        elif split == "valid":
            data_yaml['val'] = "../valid/images"
        elif split == "test":
            data_yaml['val'] = "../test/images"

        try:
            with data_path_original.open('w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.val_output.setText(f"寫入 data.yaml 失敗: {e}")
            return

        name = os.path.splitext(weights)[0] + f"_val_{split}"

        command = (
            f"yolo {model_type.lower()} val "
            f"data=\"{data_path_original}\" "
            f"model=\"{Path(self.weights_path) / weights}\" "
            f"name={name}"
        )

        self.val_output.setText(command)

    # -------------------- 預測 分頁 方法 --------------------
    def select_predict_weights_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 weights 目錄")
        if directory:
            self.predict_weights_path = directory
            self.predict_weights_display.setText(directory)
            self.load_weights_predict()

    def select_source_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇圖像來源目錄")
        if directory:
            self.source_path = directory
            self.source_display.setText(directory)

    def load_weights_predict(self):
        self.predict_selected_weights_combo.clear()
        # 檢查 predict_weights_path 是否存在且為資料夾
        if not hasattr(self, 'predict_weights_path') or not self.predict_weights_path or not os.path.isdir(self.predict_weights_path):
            self.predict_selected_weights_combo.addItem("未選擇 weights 目錄")
            return
        weights_files = [f for f in os.listdir(self.predict_weights_path) if f.endswith(".pt")]
        if not weights_files:
            self.predict_selected_weights_combo.addItem("無可用的權重檔案")
        else:
            self.predict_selected_weights_combo.addItems(weights_files)

    def generate_predict_command(self):
        source = self.source_display.text()
        model_type = self.predict_model_type_combo.currentText()
        weights = self.predict_selected_weights_combo.currentText()
        conf = "0.25"  # 默認值
        iou = "0.45"   # 默認值

        if not hasattr(self, 'predict_weights_path') or not self.predict_weights_path:
            self.predict_output.setText("請選擇 weights 目錄。")
            return

        if weights.startswith("未選擇") or weights.startswith("無可用的"):
            self.predict_output.setText("請選擇有效的權重檔案。")
            return

        if not source:
            self.predict_output.setText("請選擇圖像來源目錄。")
            return

        if not Path(source).exists():
            self.predict_output.setText("選擇的圖像來源目錄不存在。")
            return

        weights_path = Path(self.predict_weights_path) / weights
        name = Path(weights).stem + "_predict"

        if not weights_path.exists():
            self.predict_output.setText(f"找不到權重檔案在 {weights_path}。")
            return

        command = (
            f"yolo {model_type.lower()} predict "
            f"source=\"{source}\" "
            f"model=\"{weights_path}\" "
            f"name={name} "
        )

        self.predict_output.setText(command)

    # -------------------- 共用 方法 --------------------
    def copy_to_clipboard(self, text_edit):
        clipboard = QApplication.clipboard()
        clipboard.setText(text_edit.toPlainText())

    # -------------------- 設定 儲存/載入 --------------------
    def load_settings(self):
        # 訓練 Tab
        self.train_data_path = self.settings.value("train/train_data_path", "")
        if not os.path.exists(self.train_data_path):
            self.train_data_path = ""   # 路徑不存在就清空，避免報錯   
        self.train_data_display.setText(self.train_data_path)
        self.models_path = self.settings.value("train/models_path", "")
        self.models_display.setText(self.models_path)

        if self.train_data_path:
            self.load_train_datasets()
            saved_dataset = self.settings.value("train/dataset", "")
            index = self.train_data_combo.findText(saved_dataset)
            if index != -1:
                self.train_data_combo.setCurrentIndex(index)
        saved_model_type = self.settings.value("train/model_type", "Detect")
        index = self.model_type_combo.findText(saved_model_type)

        if index != -1:
            self.model_type_combo.setCurrentIndex(index)
            self.update_model_configs_train(saved_model_type)
        saved_model_config = self.settings.value("train/model_config", "")
        index = self.model_config_combo.findText(saved_model_config)

        if index != -1:
            self.model_config_combo.setCurrentIndex(index)
        self.batch_input.setText(self.settings.value("train/batch", "20"))
        self.epochs_input.setText(self.settings.value("train/epochs", "500"))
        self.imgsz_input.setText(self.settings.value("train/imgsz", "640"))
        self.train_output.setText(self.settings.value("train/command", ""))

        # 驗證 Tab
        self.val_train_data_path = self.settings.value("val/val_train_data_path", "")
        if not self.val_train_data_path or not os.path.isdir(self.val_train_data_path):
            self.val_train_data_path = ""
        self.val_train_data_display.setText(self.val_train_data_path)

        self.weights_path = self.settings.value("val/weights_path", "")
        if not self.weights_path or not os.path.isdir(self.weights_path):
            self.weights_path = ""
        self.weights_display.setText(self.weights_path)

        if self.val_train_data_path:
            self.load_val_datasets()
            saved_val_dataset = self.settings.value("val/dataset", "")
            index = self.val_data_combo.findText(saved_val_dataset)
            if index != -1:
                self.val_data_combo.setCurrentIndex(index)
        saved_val_model_type = self.settings.value("val/model_type", "Detect")
        index = self.val_model_type_combo.findText(saved_val_model_type)
        if index != -1:
            self.val_model_type_combo.setCurrentIndex(index)

        if self.weights_path:
            self.load_weights_val()
            saved_weights = self.settings.value("val/weights", "")
            index = self.selected_weights_combo.findText(saved_weights)
            if index != -1:
                self.selected_weights_combo.setCurrentIndex(index)
        saved_split = self.settings.value("val/split", "valid")
        index = self.split_combo.findText(saved_split)

        if index != -1:
            self.split_combo.setCurrentIndex(index)
        self.val_output.setText(self.settings.value("val/command", ""))

        # 預測 Tab
        self.predict_weights_path = self.settings.value("predict/predict_weights_path", "")
        if not self.predict_weights_path or not os.path.isdir(self.predict_weights_path):
            self.predict_weights_path = ""
        self.predict_weights_display.setText(self.predict_weights_path)

        if self.predict_weights_path:
            self.load_weights_predict()
            saved_predict_weights = self.settings.value("predict/weights", "")
            index = self.predict_selected_weights_combo.findText(saved_predict_weights)
            if index != -1:
                self.predict_selected_weights_combo.setCurrentIndex(index)
        self.source_path = self.settings.value("predict/source_path", "")
        self.source_display.setText(self.source_path)
        saved_predict_model_type = self.settings.value("predict/model_type", "Detect")
        index = self.predict_model_type_combo.findText(saved_predict_model_type)

        if index != -1:
            self.predict_model_type_combo.setCurrentIndex(index)
        self.predict_output.setText(self.settings.value("predict/command", ""))

        # 標註 Tab
        self.annotate_image_path = self.settings.value("annotate/image_path", "")
        if not self.annotate_image_path or not os.path.isdir(self.annotate_image_path):
            self.annotate_image_path = ""
        self.annotate_image_display.setText(self.annotate_image_path)

        if self.annotate_image_path:
            image_dir_name = os.path.basename(os.path.normpath(self.annotate_image_path))
            parent_dir = os.path.dirname(os.path.normpath(self.annotate_image_path))
            self.annotate_output_directory = os.path.join(parent_dir, f"{image_dir_name}_auto_annotate_labels")
            self.annotate_output_display.setText(self.annotate_output_directory)
        self.annotate_weights_path = self.settings.value("annotate/weights_path", "")

        if not self.annotate_weights_path or not os.path.isdir(self.annotate_weights_path):
            self.annotate_weights_path = ""
        self.annotate_weights_combo.clear()

        if self.annotate_weights_path:
            weights_files = [f for f in os.listdir(self.annotate_weights_path) if f.endswith(".pt")]
            if not weights_files:
                self.annotate_weights_combo.addItem("無可用的權重檔案")
            else:
                self.annotate_weights_combo.addItems(weights_files)
        self.annotate_conf_input.setText(self.settings.value("annotate/conf", "0.25"))
        self.annotate_iou_input.setText(self.settings.value("annotate/iou", "0.45"))
        self.annotate_output.setText(self.settings.value("annotate/output_text", ""))

        # 分配 Tab
        self.distribute_image_path = self.settings.value("distribute/image_path", "")
        if not self.distribute_image_path or not os.path.isdir(self.distribute_image_path):
            self.distribute_image_path = ""
        self.distribute_image_display.setText(self.distribute_image_path)

        self.distribute_annotation_path = self.settings.value("distribute/annotation_path", "")
        if not self.distribute_annotation_path or not os.path.isdir(self.distribute_annotation_path):
            self.distribute_annotation_path = ""
        self.distribute_annotation_display.setText(self.distribute_annotation_path)

        self.distribute_num_groups = self.settings.value("distribute/num_groups", 1, type=int)
        self.distribute_num_groups_spin.setValue(self.distribute_num_groups)
        self.distribute_output_base_dir = self.settings.value("distribute/output_base_dir", "")

        if self.distribute_output_base_dir and not os.path.isdir(self.distribute_output_base_dir):
            self.distribute_output_base_dir = ""
        self.distribute_output_display.setText(self.distribute_output_base_dir)
        self.distribute_output.setText(self.settings.value("distribute/output_text", ""))

    def save_settings(self):
        self.settings.setValue("train/train_data_path", self.train_data_path)
        self.settings.setValue("train/models_path", self.models_path)
        self.settings.setValue("train/dataset", self.train_data_combo.currentText())
        self.settings.setValue("train/model_type", self.model_type_combo.currentText())
        self.settings.setValue("train/model_config", self.model_config_combo.currentText())
        self.settings.setValue("train/batch", self.batch_input.text())
        self.settings.setValue("train/epochs", self.epochs_input.text())
        self.settings.setValue("train/imgsz", self.imgsz_input.text())
        self.settings.setValue("train/command", self.train_output.toPlainText())

        self.settings.setValue("val/val_train_data_path", self.val_train_data_path)
        self.settings.setValue("val/weights_path", self.weights_path)
        self.settings.setValue("val/dataset", self.val_data_combo.currentText())
        self.settings.setValue("val/model_type", self.val_model_type_combo.currentText())
        self.settings.setValue("val/weights", self.selected_weights_combo.currentText())
        self.settings.setValue("val/split", self.split_combo.currentText())
        self.settings.setValue("val/command", self.val_output.toPlainText())

        self.settings.setValue("predict/predict_weights_path", self.predict_weights_path)
        self.settings.setValue("predict/weights", self.predict_selected_weights_combo.currentText())
        self.settings.setValue("predict/source_path", self.source_path)
        self.settings.setValue("predict/model_type", self.predict_model_type_combo.currentText())
        self.settings.setValue("predict/command", self.predict_output.toPlainText())

        self.settings.setValue("annotate/image_path", self.annotate_image_path)
        self.settings.setValue("annotate/weights_path", self.annotate_weights_path)
        self.settings.setValue("annotate/conf", self.annotate_conf_input.text())
        self.settings.setValue("annotate/iou", self.annotate_iou_input.text())
        self.settings.setValue("annotate/output_text", self.annotate_output.toPlainText())

        self.settings.setValue("distribute/image_path", self.distribute_image_path)
        self.settings.setValue("distribute/annotation_path", self.distribute_annotation_path)
        self.settings.setValue("distribute/num_groups", self.distribute_num_groups_spin.value())
        self.settings.setValue("distribute/output_base_dir", self.distribute_output_base_dir)
        self.settings.setValue("distribute/output_text", self.distribute_output.toPlainText())

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    window = YOLOTrainerValidatorPredictor()
    window.show()
    sys.exit(app.exec())
