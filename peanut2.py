from ultralytics import YOLO
import torch
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label
from skimage.segmentation import watershed
from torchviz import make_dot
import netron
from torchsummary import summary





def export_model_to_onnx(model, export_path="model.onnx", input_size=(1, 3, 640, 640)):
    """
    Export the trained YOLO model to the ONNX format and save it to the specified path.
    
    Args:
        model: The trained PyTorch model.
        export_path (str): The file path where the model should be saved.
        input_size (tuple): The input size for the dummy tensor (batch_size, channels, height, width).
    """
    # Create a dummy input tensor with the specified input size (for example, 1 batch of 3x640x640 image)
    dummy_input = torch.randn(*input_size)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Export the model to ONNX format
    torch.onnx.export(model.model, dummy_input, export_path, opset_version=11)
    
    print(f"Model exported successfully to {export_path}")

def visualize_onnx_model(export_path="model.onnx"):
    """
    Visualize the exported ONNX model using Netron.
    
    Args:
        export_path (str): The file path to the ONNX model to visualize.
    """
    # Start the Netron server and visualize the model
    netron.start(export_path)
    print(f"Visualizing model: {export_path} using Netron.")

def train_model():
    """Train the YOLO model and then export and visualize the ONNX model."""
    # Choose the device (cuda if available, else cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the YOLO model (replace with your model path if necessary)
    model_path = r"I:\\WLB\\YOLO11\\ultralytics\\runs\\weights\\best.pt"
    model = YOLO(model_path)
    
    # Train the YOLO model
    train_results = model.train(
        data=r"I:\\WLB\\YOLO11\\ultralytics\\roboflow\\data.yaml",  # Dataset YAML file
        epochs=10,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=device,  # Automatically choose device based on availability
        workers=0  # # Reduce number of workers to prevent file access issues
    )
    print(f"Training results: {train_results}")
    
    # Optionally, validate the model after training
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    # Export the trained model to ONNX format
    export_model_to_onnx(model, export_path="trained_model.onnx", input_size=(1, 3, 640, 640))
    
    # Visualize the exported ONNX model using Netron
    visualize_onnx_model(export_path="trained_model.onnx")


def analyze_and_visualize_images(image_folder, model):
    """
    Analyze images in a folder and visualize detected instances with unique colors.
    
    Args:
        image_folder (str): Path to the folder containing images.
        model (YOLO): Pretrained YOLO model for prediction.
    """
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            original_image = cv2.imread(image_path)
            original_height, original_width = original_image.shape[:2]

            # Predict using YOLO
            results = model.predict(source=image_path)

            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    scale = min(640 / original_width, 640 / original_height)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    pad_w = (640 - new_width) // 2
                    pad_h = (640 - new_height) // 2

                    masks = result.masks.data.cpu().numpy()
                    bboxes = result.boxes.xyxy.cpu().numpy()

                    adjusted_masks = []
                    for mask in masks:
                        cropped_mask = mask[pad_h:pad_h + new_height, pad_w:pad_w + new_width]
                        resized_mask = cv2.resize(cropped_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                        adjusted_masks.append(resized_mask)

                    colored_masks = np.zeros_like(original_image, dtype=np.uint8)
                    np.random.seed(42)  # For reproducible random colors
                    colors = [np.random.randint(0, 255, size=3, dtype=np.uint8) for _ in range(len(adjusted_masks))]

                    for i, mask in enumerate(adjusted_masks):
                        colored_masks[mask > 0] = colors[i]

                    blended_image = cv2.addWeighted(original_image, 0.6, colored_masks, 0.4, 0)

                    # Draw bounding boxes on the blended image
                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox
                        x_min = max(0, int((x_min - pad_w) / scale))
                        y_min = max(0, int((y_min - pad_h) / scale))
                        x_max = min(original_width, int((x_max - pad_w) / scale))
                        y_max = min(original_height, int((y_max - pad_h) / scale))
                        cv2.rectangle(blended_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Red for boxes

                    # Display the results
                    plt.figure(figsize=(10, 10))
                    plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Processed Image: {filename}")
                    plt.axis("off")
                    plt.show()

if __name__ == "__main__":

    # Define the folder containing images and the output folder
    image_folder = r"I:\\WLB\\YOLO11\\ultralytics\\roboflow\\test\\ts"  # Folder with images
    output_folder = r"I:\\WLB\\YOLO11\\ultralytics\\roboflow\\test\\images\\labeled_results"  # Folder to save results
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    # Load the YOLO model
    model_path = r"I:\\WLB\\YOLO11\\ultralytics\\runs\\weights\\best.pt"
    model = YOLO(model_path)

    # Define grades
    grades = {
        0: (0, 0, "Grade 0"),
        1: (0, 10, "Grade 1"),
        2: (10, 20, "Grade 2"),
        3: (20, 50, "Grade 3"),
        4: (50, 80, "Grade 4"),
        5: (80, 100, "Grade 5")
    }

    # Generate a color map from Matplotlib's 'cool' palette
    colormap = plt.cm.get_cmap('cool', 6)
    colormap = [tuple(int(c * 255) for c in colormap(i)[:3]) for i in range(6)]


    analyze_and_visualize_images(image_folder, model)

