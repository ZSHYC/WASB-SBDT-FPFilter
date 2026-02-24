from ultralytics import YOLO
import argparse
import os
import numpy as np


def predict_image(model, output_folder, dirname, image_name, conf_thresh=0.5):
    image_path = os.path.join(dirname, image_name)
    # if image_name < "1755226641631.jpg":
    #     return 
    # Run prediction on the image
    results = model(image_path)

    # Get the bounding boxes and class information
    boxes = results[0].boxes.xywhn  # Get bounding box
    classes = results[0].boxes.cls  # Get class labels
    confidences = results[0].boxes.conf  # Get confidence scores

    # Save the result label file in YOLO format
    output_label_path = os.path.join(output_folder, image_name.replace('.jpg', '.txt'))
    # for box, cls, conf in zip(boxes, classes, confidences):
    #     print(f"conf: {conf} {image_name} {cls} {box}")

    # # # TODO don't remove thiese
    # ball_count = 0 
    # for box, cls, conf in zip(boxes, classes, confidences):
    #     print(f"conf: {conf} {image_name}")
    #     if conf > 0.4:
    #         ball_count += 1
    # if ball_count > 5:
    #     return 
    
    with open(output_label_path, 'w') as f:
        for box, cls, conf in zip(boxes, classes, confidences):
            print(f"conf: {conf} {image_name}")
            if conf > conf_thresh:
                x, y, w, h = box
                f.write(f"{int(cls)} {x} {y} {w} {h}\n")


def predict_folder(model, input_folder, output_folder, conf_thresh=0.5):
    """Run predictions on all images in input_folder using provided model and save labels to output_folder."""
    os.makedirs(output_folder, exist_ok=True)

    image_list = []
    for image_name in os.listdir(input_folder):
        lower = image_name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            image_list.append(image_name)

    image_list.sort()

    for image_name in image_list:
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            try:
                predict_image(model, output_folder, input_folder, image_name, conf_thresh=conf_thresh)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")


def load_model(model_path):
    """Load and return a YOLO model from model_path."""
    return YOLO(model_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save results")
    parser.add_argument("--model", type=str, default="yolov8n_1280_1113.pt", help="Path to model file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    model = load_model(args.model)
    predict_folder(model, args.input_folder, args.output_folder, conf_thresh=args.conf)
# python yolo_predict.py --input_folder /home/zrgy/datasets/serve_images --output_folder /home/zrgy/datasets/serve_images_yolo_txt
# python yolo_predict.py --input_folder ./right_frames --output_folder ./right_labels
