import sys
import os
import warnings
import copy
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import (GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, 
                              HiResCAM, LayerCAM, RandomCAM, EigenGradCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients as BaseActivationsAndGradients

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QComboBox, QFileDialog, QCheckBox, QDoubleSpinBox, 
                             QSpinBox, QStatusBar, QFormLayout)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
np.random.seed(0)

# 調整圖像大小與補邊，確保圖像符合模型輸入要求
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)

# 處理激活值與梯度資訊的類別 (用於 Grad-CAM 後處理)
class ActivationsAndGradients(BaseActivationsAndGradients):
    # 對模型輸出進行後處理，排序並選取需要的信息
    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

# YOLO 偵測任務後處理模組，依據設定取出所需輸出 (類別與邊框)
class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        # 初始化偵測目標參數
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
    
    def forward(self, data):
        # 處理偵測結果 (類別與邊框)
        post_result, pre_post_boxes = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result) if result else torch.tensor(0.0)

# YOLO 分割任務後處理模組 (額外處理 mask)
class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        # 處理分割結果 (包含 mask 數據)
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result) if result else torch.tensor(0.0)

# YOLO 姿態估計任務後處理模組
class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        # 處理姿態估計結果 (輸出 box 與姿態資訊)
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result) if result else torch.tensor(0.0)

# YOLO 旋轉邊界框任務後處理模組 (處理角度資訊)
class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        # 處理旋轉邊界框結果 (包含角度)
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result) if result else torch.tensor(0.0)

# YOLO 分類任務後處理模組
class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        # 直接回傳分類結果
        return data.max()

# 生成 YOLO 熱力圖模組，整合模型、Grad-CAM 與後處理
class YoloHeatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size, status_callback=None):
        # 設定回呼函式，用於傳遞狀態訊息給 GUI 或終端
        self.status_callback = status_callback
        
        self.log(f'正在載入模型: {weight}...')
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        self.log(f'模型類別資訊: {model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info(verbose=False) # 減少終端輸出
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False
        
        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"不支援的任務({task})")
        
        target_layers = [model.model[l] for l in layer]
        method_class = eval(method)
        cam_method = method_class(model, target_layers)
        cam_method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        self.__dict__.update(locals())

    # log 函式：將訊息輸出至 GUI 狀態欄或終端
    def log(self, message):
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    # 將 CAM 限定在偵測框內，重新標準化 CAM
    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        if boxes.shape[0] > 0:
            for x1, y1, x2, y2 in boxes:
                x1, y1 = max(int(x1), 0), max(int(y1), 0)
                x2, y2 = min(int(x2), grayscale_cam.shape[1] - 1), min(int(y2), grayscale_cam.shape[0] - 1)
                if x1 < x2 and y1 < y2: # 確保框有效
                    renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    # 處理單張圖像，生成熱力圖與偵測結果圖像
    def process_image(self, img_path, output_dir):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_subdir = os.path.join(output_dir, base_name)
        os.makedirs(save_subdir, exist_ok=True)
        
        heatmap_save_path = os.path.join(save_subdir, 'heatmap.png')
        detection_save_path = os.path.join(save_subdir, 'detection.png')
        
        self.log(f"處理中: {os.path.basename(img_path)}")

        try:
            img_raw = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            if img_raw is None:
                raise IOError(f"無法讀取圖像: {img_path}")
        except Exception as e:
            self.log(f"警告... 讀取 {img_path} 失敗: {e}")
            return
        
        img, _, (top, bottom, left, right) = letterbox(img_raw, new_shape=(self.img_size, self.img_size), auto=True)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        try:
            grayscale_cam = self.cam_method(tensor, [self.target])
        except Exception as e:
            self.log(f"警告... 計算CAM失敗於 {img_path}: {e}")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        pred_results = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
        
        # 儲存偵測結果圖
        detection_plot = pred_results.plot(img=img_rgb.copy()) # 使用原始RGB圖像繪製
        detection_plot_bgr = cv2.cvtColor(detection_plot, cv2.COLOR_RGB2BGR)
        detection_plot_bgr = detection_plot_bgr[top:detection_plot_bgr.shape[0] - bottom, left:detection_plot_bgr.shape[1] - right]
        cv2.imencode('.png', detection_plot_bgr)[1].tofile(detection_save_path)

        # 處理熱力圖
        if self.renormalize and self.task in ['detect', 'segment', 'pose']:
            boxes_xyxy = pred_results.boxes.xyxy.cpu().detach().numpy()
            cam_image = self.renormalize_cam_in_bounding_boxes(boxes_xyxy, img_float, grayscale_cam)
        
        if self.show_result:
            cam_image = pred_results.plot(img=cam_image, conf=True, line_width=None, labels=False)
        
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image_pil = Image.fromarray(cam_image)
        cam_image_pil.save(heatmap_save_path)

# --- PySide6 GUI 程式碼 ---

# 背景工作執行緒模組，用於處理耗時的圖像辨識與生成任務
class Worker(QThread):
    """
    背景工作執行緒，用於處理耗時的圖像辨識任務
    """
    status_update = Signal(str)
    finished = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_stopped = False
        self.heatmap_model = None

    def run(self):
        # 執行背景任務，逐張處理資料夾內的圖像
        try:
            self.status_update.emit("正在初始化模型...")
            self.heatmap_model = YoloHeatmap(
                **self.params['model_params'],
                status_callback=self.status_update.emit
            )
            
            input_dir = self.params['input_dir']
            output_dir = self.params['output_dir']
            
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
            total_images = len(image_files)
            
            for i, filename in enumerate(image_files):
                if self.is_stopped:
                    self.status_update.emit("任務已由使用者手動停止。")
                    break
                
                self.status_update.emit(f"進度: {i+1}/{total_images} - 正在處理 {filename}")
                img_path = os.path.join(input_dir, filename)
                self.heatmap_model.process_image(img_path, output_dir)
            
            if not self.is_stopped:
                self.status_update.emit("所有圖像處理完成！")

        except Exception as e:
            self.status_update.emit(f"發生錯誤: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        # 設定停止旗標，停止執行緒操作
        self.is_stopped = True

# 主視窗模組，使用 PySide6 建立圖形介面
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 設定視窗標題與初始尺寸
        self.setWindowTitle("YOLO 熱力圖與辨識結果生成器")
        self.setGeometry(100, 100, 700, 600)

        # 主要元件與版面配置
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        form_layout = QFormLayout()

        # 輸入與輸出路徑
        self.input_dir_line = QLineEdit()
        self.input_dir_line.setText(r"data")  # 預設輸入資料夾設定為 data 資料夾
        self.output_dir_line = QLineEdit()
        self.output_dir_line.setText(r"run")  # 預設輸出資料夾設定為 run 資料夾
        self.weight_path_line = QLineEdit()

        btn_input = QPushButton("選擇資料夾")
        btn_output = QPushButton("選擇儲存位置")
        btn_weight = QPushButton("選擇權重檔案")
        
        btn_input.clicked.connect(lambda: self.select_directory(self.input_dir_line))
        btn_output.clicked.connect(lambda: self.select_directory(self.output_dir_line))
        btn_weight.clicked.connect(lambda: self.select_file(self.weight_path_line, "PyTorch Models (*.pt)"))

        form_layout.addRow("輸入圖像資料夾:", self.create_path_widget(self.input_dir_line, btn_input))
        form_layout.addRow("輸出結果資料夾:", self.create_path_widget(self.output_dir_line, btn_output))
        form_layout.addRow("模型權重檔案:", self.create_path_widget(self.weight_path_line, btn_weight))
        
        # 預設參數
        self.method_combo = QComboBox()
        self.method_combo.addItems(['GradCAMPlusPlus', 'GradCAM', 'XGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'RandomCAM', 'EigenGradCAM'])
        form_layout.addRow("熱力圖方法 (Method):", self.method_combo)
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(['detect', 'segment', 'pose', 'obb', 'classify'])
        form_layout.addRow("模型任務 (Task):", self.task_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda:0', 'cpu'] if torch.cuda.is_available() else ['cpu'])
        form_layout.addRow("計算裝置 (Device):", self.device_combo)

        self.layer_line = QLineEdit("10, 12, 14, 16, 18")
        form_layout.addRow("目標層 (Layers, comma-separated):", self.layer_line)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.2)
        self.conf_spin.setSingleStep(0.05)
        form_layout.addRow("信心度閾值 (Conf Threshold):", self.conf_spin)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.0, 1.0)
        self.ratio_spin.setValue(0.02)
        self.ratio_spin.setSingleStep(0.01)
        form_layout.addRow("目標比例 (Ratio):", self.ratio_spin)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 2048)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        form_layout.addRow("圖像尺寸 (Image Size):", self.img_size_spin)

        self.show_result_check = QCheckBox()
        self.show_result_check.setChecked(True)
        form_layout.addRow("在熱力圖上顯示辨識結果:", self.show_result_check)

        self.renormalize_check = QCheckBox()
        self.renormalize_check.setChecked(False)
        form_layout.addRow("將熱力圖限制在框內 (Renormalize):", self.renormalize_check)

        # 按鈕區域
        self.start_button = QPushButton("開始處理")
        self.stop_button = QPushButton("停止")
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # 狀態列
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.worker = None

    def create_path_widget(self, line_edit, button):
        # 建立路徑選擇元件 (文字欄位與按鈕)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return widget

    def select_directory(self, line_edit):
        # 開啟資料夾選擇對話框
        path = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if path:
            line_edit.setText(path)

    def select_file(self, line_edit, filter_str):
        # 開啟檔案選擇對話框
        path, _ = QFileDialog.getOpenFileName(self, "選擇檔案", "", filter_str)
        if path:
            line_edit.setText(path)

    def start_processing(self):
        # 收集參數並啟動背景處理執行緒
        input_dir = self.input_dir_line.text()
        output_dir = self.output_dir_line.text()
        weight_path = self.weight_path_line.text()
        
        if not all([input_dir, output_dir, weight_path]):
            self.status_bar.showMessage("錯誤: 請填寫所有路徑！")
            return
        if not os.path.isdir(input_dir):
            self.status_bar.showMessage("錯誤: 輸入資料夾不存在！")
            return
        if not os.path.isfile(weight_path):
            self.status_bar.showMessage("錯誤: 權重檔案不存在！")
            return
        
        os.makedirs(output_dir, exist_ok=True)

        try:
            layers = [int(l.strip()) for l in self.layer_line.text().split(',')]
        except ValueError:
            self.status_bar.showMessage("錯誤: Layer 格式不正確，請使用逗號分隔的數字。")
            return

        model_params = {
            'weight': weight_path,
            'device': self.device_combo.currentText(),
            'method': self.method_combo.currentText(),
            'layer': layers,
            'backward_type': 'all', # 根據原碼固定為 'all'
            'conf_threshold': self.conf_spin.value(),
            'ratio': self.ratio_spin.value(),
            'show_result': self.show_result_check.isChecked(),
            'renormalize': self.renormalize_check.isChecked(),
            'task': self.task_combo.currentText(),
            'img_size': self.img_size_spin.value()
        }
        
        all_params = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'model_params': model_params
        }

        # 2. 建立並啟動 Worker
        self.worker = Worker(all_params)
        self.worker.status_update.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

        # 3. 更新 GUI 狀態
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_processing(self):
        # 發送停止訊號給背景執行緒
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.status_bar.showMessage("正在傳送停止訊號...")

    def on_processing_finished(self):
        # 處理完成後更新 GUI 狀態
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    def closeEvent(self, event):
        # 關閉視窗前確保背景執行緒正常停止
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait() # 等待執行緒結束
        event.accept()


if __name__ == '__main__':
    # 程式入口，建立並執行應用程式
    app = QApplication(sys.argv)
    
    # 設置全局字體大小
    font = QFont()
    font.setPointSize(12)  # 您可以根據需要調整這個數值
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())