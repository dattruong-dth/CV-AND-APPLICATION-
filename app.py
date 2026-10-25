import gradio as gr
import cv2
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.applications.resnet50 import preprocess_input as preprocess_resnet

import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    os.system('pip install -q efficientnet_pytorch')
    from efficientnet_pytorch import EfficientNet


PATHS = {
    "detect": {
        "YOLOv8": r"detect/Yolov8/newf/results (1)/runs/detect/face_detect_success/weights/best.pt",
        "YOLO11": r"detect/yolo11/runs/detect/train6/weights/best.pt"
    },
    "attribute": {
        "ResNet50 (Keras)": r"attribute/resnet/Final_ResNet50_Transfer.keras",
        "MobileNet (Keras)": r"attribute/mobilenet/models/MobileNetV3_Fixed_10Epochs.keras",
        "EfficientNet-B4 (PyTorch)": r"attribute/efficientnet/tmp/models/best_efficientnet_weighted.pth"
    }
}

ATTR_LABELS = [
    'Male', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
    'Straight_Hair', 'Wavy_Hair', 'Receding_Hairline', 'Mustache', 'Sideburns',
    'No_Beard', 'Eyeglasses', 'Wearing_Hat', 'Wearing_Earrings', 'Wearing_Necklace',
    'Wearing_Necktie'
]

IMG_SIZE = (224, 224) 

def load_efficientnet_pytorch(weights_path):
    try:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=len(ATTR_LABELS))
        
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        threshold_path = os.path.join(os.path.dirname(weights_path), 'best_thresholds.npy')
        
        if os.path.exists(threshold_path):
            thresholds = np.load(threshold_path)
            print(f"✅ Đã load bộ ngưỡng tối ưu từ {threshold_path}")
        else:
            print(f"⚠️ Không tìm thấy {threshold_path}, sử dụng ngưỡng mặc định 0.5")
            thresholds = np.ones(len(ATTR_LABELS)) * 0.5

        return model, thresholds, None
        
    except Exception as e:
        return None, None, str(e)


def preprocess_image_torch(img_arr):
    # Logic này copy từ class CelebDataset trong code train của bạn
    
    # 1. Resize
    img = cv2.resize(img_arr, IMG_SIZE)
    
    # 2. Normalize theo chuẩn ImageNet (quan trọng!)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # 3. Chuyển kênh màu (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    
    # 4. Tạo Batch dimension
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return tensor


def preprocess_image_keras(img_arr, model_name):
    # Resize về 224x224
    img = cv2.resize(img_arr, IMG_SIZE)
    img = img.astype('float32')
    
    if "ResNet" in model_name:
        img = np.expand_dims(img, axis=0)
        img = preprocess_resnet(img)
    else:
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
    return img

def expand_box(x1, y1, x2, y2, img_w, img_h, ratio=0.0):
    if ratio <= 0: return x1, y1, x2, y2
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * ratio)
    pad_y = int(h * ratio) 
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y) 
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y) 
    return nx1, ny1, nx2, ny2

# --- PIPELINE CHÍNH ---
def run_pipeline(input_image, detect_model_name, attr_model_name, padding_ratio):
    if input_image is None:
        return None, "Vui lòng upload ảnh."

    # --- BƯỚC 1: DETECTION ---
    detect_path = PATHS['detect'][detect_model_name]
    if not os.path.exists(detect_path):
        return input_image, f"Error: Không tìm thấy file detect tại {detect_path}"
    
    try:
        yolo_model = YOLO(detect_path)
        results = yolo_model(input_image)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    except Exception as e:
        return input_image, f"Lỗi chạy YOLO: {e}"
    
    if len(boxes) == 0:
        return input_image, "Không phát hiện khuôn mặt nào."

    # --- BƯỚC 2: LOAD MODEL ATTRIBUTE ---
    attr_path = PATHS['attribute'][attr_model_name]
    if not os.path.exists(attr_path):
        return input_image, f"Error: Không tìm thấy file attribute tại {attr_path}"

    clf_model = None
    thresholds = None 
    framework = "keras"
    
    if "EfficientNet" in attr_model_name:
        # Load Model PyTorch + Thresholds
        clf_model, thresholds, error_msg = load_efficientnet_pytorch(attr_path)
        framework = "torch"
        if clf_model is None: 
            return input_image, f"Lỗi load EfficientNet:\n{error_msg}"
    else:
        # Load Model Keras
        try:
            clf_model = keras.models.load_model(attr_path, compile=False)
        except Exception as e:
            return input_image, f"Lỗi load Keras model: {e}"

    output_img = input_image.copy()
    img_h, img_w = input_image.shape[:2]
    final_report = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        nx1, ny1, nx2, ny2 = expand_box(x1, y1, x2, y2, img_w, img_h, padding_ratio)
        cv2.rectangle(output_img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
        cv2.putText(output_img, f"#{i+1}", (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        face_crop = input_image[ny1:ny2, nx1:nx2]
        if face_crop.size == 0: continue

        preds = []
        try:
            if framework == "keras":
                # Keras (ResNet/MobileNet)
                inp = preprocess_image_keras(face_crop, attr_model_name)
                out = clf_model.predict(inp, verbose=0)
                if isinstance(out, list): out = out[0]
                preds = np.array(out).flatten()
                # Keras model mặc định threshold là 0.5
                current_thresholds = np.ones(len(ATTR_LABELS)) * 0.5
                
            else: 
                inp = preprocess_image_torch(face_crop)
                with torch.no_grad():
                    out = clf_model(inp)
                    out = torch.sigmoid(out) 
                    preds = out.cpu().numpy()[0]
                current_thresholds = thresholds
                
        except Exception as e:
            final_report.append(f"**Face #{i+1}:** Lỗi dự đoán ({e})")
            continue


        attr_scores = []
        for idx, score in enumerate(preds):
            thresh = current_thresholds[idx]
            is_active = score > thresh
            attr_scores.append((ATTR_LABELS[idx], score, is_active))
        
        attr_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        report_lines = [f"### Face #{i+1} Results:"]
        for label, score, is_active in attr_scores:
            if is_active:
                # In đậm nếu vượt qua ngưỡng
                report_lines.append(f"- **{label}: {score:.4f}**")
            else:
                report_lines.append(f"- {label}: {score:.4f}")
        
        final_report.append("\n".join(report_lines))
        final_report.append("---")

    return output_img, "\n".join(final_report)


with gr.Blocks(title="Face Analysis System") as demo:
    gr.Markdown("## Hệ thống Detect & Phân loại Khuôn mặt")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ảnh đầu vào", type="numpy")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Cấu hình Model")
                detect_dd = gr.Dropdown(
                    choices=list(PATHS['detect'].keys()), 
                    value="YOLOv8", 
                    label="Model Detect"
                )
                attr_dd = gr.Dropdown(
                    choices=list(PATHS['attribute'].keys()), 
                    value="EfficientNet-B4 (PyTorch)", 
                    label="Model Attribute"
                )
                pad_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Padding Ratio (Mở rộng Box)",
                )
            
            btn = gr.Button("🚀 Phân tích ngay", variant="primary")
        
        with gr.Column(scale=1):
            output_img = gr.Image(label="Kết quả")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Bảng Chi tiết")

    btn.click(
        run_pipeline, 
        inputs=[input_img, detect_dd, attr_dd, pad_slider], 
        outputs=[output_img, output_text]
    )

if __name__ == "__main__":
    demo.launch()