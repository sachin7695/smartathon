import streamlit as st
import torch
import random
import cv2
from PIL import Image
from pathlib import Path
import os 

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.datasets import LoadImages
from yolov7.utils.torch_utils import select_device
from yolov7.utils.plots import plot_one_box
from yolov7.utils.general import non_max_suppression, set_logging, scale_coords

def detect(source, imgsz, weights, classes=None, conf_thres=0.25, iou_thres=0.45, agnostic_nms=False):
    # Initialize
    set_logging()
    device = select_device('cpu')

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img_output = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for det in pred:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            img_output.append(im0)
    print('Done')
    return img_output

st.title('Smartathon')

uploaded_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    path = os.path.join("", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(bytes_data)
    st.image(path, caption='Uploaded Image', use_column_width=True)    
    
    if st.button('Predict'):
        with torch.no_grad():
            img_output = detect(path, 640, 'best.pt')
        if img_output:
            st.image(img_output[0], caption='Predicted Image', use_column_width=True)