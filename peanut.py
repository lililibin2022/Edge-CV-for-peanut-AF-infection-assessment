from ultralytics import YOLO
import torch
from pathlib import Path
import os

def train_model():
    # Load the YOLO model
    model = YOLO("yolo11s.pt")

    # Determine device (cuda or cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train the model
    train_results = model.train(
        data=r"I:\WLB\YOLO11\ultralytics\roboflow\data.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device=device  # automatically choose device
    )

    # Print training results
    print(f"Training results: {train_results}")

    # Evaluate model performance on the validation set
    metrics = model.val()
    print(f"Validation metrics: {metrics}")

    # Perform object detection on images
    image_directory = r"I:\WLB\YOLO11\ultralytics\roboflow\test\images"
    output_directory = r"I:\WLB\YOLO11\ultralytics\roboflow\test\output_images"

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process images
    for image_path in Path(image_directory).glob("*.[jp][pn][g]*"):  # Flexible format matching
        print(f"Processing image: {image_path}")
        results = model(image_path)  # Predict on the image

        # Iterate through each result and save individually
        for i, result in enumerate(results):
            result_image_path = Path(output_directory) / f"output_{image_path.stem}_{i}.jpg"
            result.save(save_dir=output_directory)  # Save all results to the directory

    # Export the model to ONNX format
    export_path = model.export(format="onnx")
    print(f"Model exported to: {export_path}")





def train_and_export():
    # Load a lightweight model (small, nano, or other suitable type)
    model = YOLO("yolo11n.pt")  # Replace with 'yolo11s.pt' or 'R18' as needed

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train the model
    train_results = model.train(
        data=r"I:\WLB\YOLO11\ultralytics\roboflow\data.yaml",  # path to dataset YAML,  # Adjust to your dataset's YAML file
        epochs=50,          # Reduce epochs for faster training in edge scenarios
        imgsz=640,
        workers=0
    )

    print(f"Training Results: {train_results}")

    # Validate the model
    metrics = model.val()
    print(f"Validation Metrics: {metrics}")

    # Export the model to ONNX for edge deployment
    export_path = model.export(format="onnx")
    print(f"Model exported to ONNX: {export_path}")

    # Optional: Quantize the ONNX model for deployment
    quantized_model_path = quantize_model(export_path)
    print(f"Quantized Model Path: {quantized_model_path}")

def quantize_model(onnx_path):
    """
    Example function for model quantization.
    Converts ONNX model to FP16 or INT8 for optimized inference on edge devices.
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8  # For INT8 quantization
    )
    return quantized_path

if __name__ == '__main__':
    train_and_export()


