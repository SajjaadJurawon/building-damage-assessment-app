from pipeline import create_yolo_pipeline_from_saved_models
from tifffile import imread
import numpy as np

# Create pipeline
pipeline = create_yolo_pipeline_from_saved_models(
    yolo_model_path=r'C:\Users\sajja\Desktop\DS\Dissertation\Project\best_yolo_50epochs.pt',
    classification_model_path=r'C:\Users\sajja\Desktop\DS\Dissertation\Project\efficientnet_stage3.pth',
    device='cuda',
    confidence_threshold=0.1
)

# Load and test image
image_path = r"C:\Users\sajja\Desktop\DS\Dissertation\Project\hurricane-harvey_00000485_post_disaster.tif"
image = imread(image_path)

if image.dtype != np.uint8:
    image = (image / image.max() * 255).clip(0, 255).astype(np.uint8)

# Run pipeline
results = pipeline.process_image(image)

# Visualize
pipeline.visualize_results(results)

# Test JSON output
import json

json_results = pipeline.results_to_json(results)
print("\n" + "="*50)
print("JSON OUTPUT:")
print("="*50)
print(json.dumps(json_results, indent=2))

# Optional: save to file to see the structure
with open('test_output.json', 'w') as f:
    json.dump(json_results, f, indent=2)
print("\nJSON saved to test_output.json")