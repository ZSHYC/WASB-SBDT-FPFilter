from ultralytics import YOLO
import argparse
import os
import numpy as np
import cv2


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


def visualize_prediction(image_path, output_image_path, boxes, classes, confidences, conf_thresh=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_h, img_w = image.shape[:2]

    for box, cls, conf in zip(boxes, classes, confidences):
        if conf <= conf_thresh:
            continue

        x, y, w, h = box
        x, y, w, h = float(x), float(y), float(w), float(h)

        x1 = int((x - w / 2.0) * img_w)
        y1 = int((y - h / 2.0) * img_h)
        x2 = int((x + w / 2.0) * img_w)
        y2 = int((y + h / 2.0) * img_h)

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)} {float(conf):.2f}"
        cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, image)


def images_to_video(image_folder, output_video_path, fps=30):
    """Convert images in image_folder (sorted) into a video saved at output_video_path."""
    images = [p for p in os.listdir(image_folder) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise ValueError(f"No images found in {image_folder}")

    images.sort()
    first_path = os.path.join(image_folder, images[0])
    first_img = cv2.imread(first_path)
    if first_img is None:
        raise ValueError(f"Failed to read first image: {first_path}")

    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: failed to read {img_path}, skipping")
            continue
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()


def predict_folder(model, input_folder, output_folder, conf_thresh=0.5, visualize=False, vis_output_folder=None):
    """Run predictions on all images in input_folder using provided model and save labels to output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    if visualize:
        if vis_output_folder is None:
            vis_output_folder = output_folder + "_vis"
        os.makedirs(vis_output_folder, exist_ok=True)

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
                results = model(image_path)

                boxes = results[0].boxes.xywhn
                classes = results[0].boxes.cls
                confidences = results[0].boxes.conf

                output_label_path = os.path.join(output_folder, image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                with open(output_label_path, 'w') as f:
                    for box, cls, conf in zip(boxes, classes, confidences):
                        print(f"conf: {conf} {image_name}")
                        if conf > conf_thresh:
                            x, y, w, h = box
                            f.write(f"{int(cls)} {x} {y} {w} {h}\n")

                if visualize:
                    output_image_path = os.path.join(vis_output_folder, image_name)
                    visualize_prediction(
                        image_path=image_path,
                        output_image_path=output_image_path,
                        boxes=boxes,
                        classes=classes,
                        confidences=confidences,
                        conf_thresh=conf_thresh,
                    )
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
    parser.add_argument("--visualize", action="store_true", help="Whether to save visualization images with predicted boxes")
    parser.add_argument("--vis_output_folder", type=str, default=None, help="Path to save visualization images (default: output_folder + '_vis')")
    parser.add_argument("--vis_to_video", action="store_true", help="Whether to convert visualized images into a video")
    parser.add_argument("--video_name", type=str, default="vis_output.mp4", help="Output video file name (or path) when --vis_to_video is set")
    parser.add_argument("--fps", type=int, default=30, help="FPS for the output video")
    args = parser.parse_args()

    model = load_model(args.model)
    predict_folder(
        model,
        args.input_folder,
        args.output_folder,
        conf_thresh=args.conf,
        visualize=args.visualize,
        vis_output_folder=args.vis_output_folder,
    )
    # If requested, convert visualized images to video
    if args.visualize and args.vis_to_video:
        vis_folder = args.vis_output_folder if args.vis_output_folder is not None else args.output_folder + "_vis"
        try:
            images_to_video(vis_folder, args.video_name, fps=args.fps)
            print(f"Saved visualization video to {args.video_name}")
        except Exception as e:
            print(f"Failed to create video from {vis_folder}: {e}")
# Examples:
# 1) Only output labels (YOLO txt):
#    python yolo_predict.py --input_folder right_yuedong_frames --output_folder right_labels_pred
# 2) Output labels and save visualized images:
#    python yolo_predict.py --input_folder eg_left --output_folder eg_left_labels_pred --visualize --vis_output_folder eg_left_vis_pred
# 3) Output labels, save visualized images and convert them to a video (specify fps and video name):
#    python yolo_predict.py --input_folder left_up --output_folder left_up_labels_pred --visualize --vis_output_folder left_up_vis_pred --vis_to_video --video_name left_up_vis.mp4 --fps 25
