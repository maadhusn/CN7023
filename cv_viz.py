"""OpenCV visualization utilities for prediction overlays."""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse


def draw_prediction_overlay(image_path, model, class_names, device, transform, save_path):
    """Draw prediction overlay on image using OpenCV."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
    
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    height, width = cv_image.shape[:2]
    if width > 800:
        scale = 800 / width
        new_width = 800
        new_height = int(height * scale)
        cv_image = cv2.resize(cv_image, (new_width, new_height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    pred_text = f"Prediction: {predicted_class}"
    conf_text = f"Confidence: {confidence_score:.2f}"
    
    (pred_w, pred_h), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
    
    max_width = max(pred_w, conf_w)
    total_height = pred_h + conf_h + 30
    
    cv2.rectangle(cv_image, (10, 10), (max_width + 20, total_height + 10), (0, 0, 0), -1)
    cv2.rectangle(cv_image, (10, 10), (max_width + 20, total_height + 10), (255, 255, 255), 2)
    
    cv2.putText(cv_image, pred_text, (15, 35), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(cv_image, conf_text, (15, 65), font, font_scale, (0, 255, 255), thickness)
    
    if confidence_score > 0.8:
        color = (0, 255, 0)  # Green for high confidence
    elif confidence_score > 0.5:
        color = (0, 255, 255)  # Yellow for medium confidence
    else:
        color = (0, 0, 255)  # Red for low confidence
    
    cv2.circle(cv_image, (max_width + 5, 25), 8, color, -1)
    
    cv2.imwrite(save_path, cv_image)


def generate_prediction_visualizations(model, test_loader, class_names, device, transform, num_samples=12, save_dir='results'):
    """Generate prediction visualizations with OpenCV overlays."""
    model.eval()
    
    samples_generated = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            probabilities = F.softmax(output, dim=1)
            
            for i in range(data.size(0)):
                if samples_generated >= num_samples:
                    break
                
                img_tensor = data[i].cpu()
                
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img_tensor = img_tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                
                temp_path = os.path.join(save_dir, f'temp_viz_{samples_generated}.jpg')
                img_pil.save(temp_path)
                
                save_path = os.path.join(save_dir, f'viz_pred_{samples_generated+1:02d}.png')
                
                from torchvision import transforms
                simple_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                draw_prediction_overlay(temp_path, model, class_names, device, simple_transform, save_path)
                
                os.remove(temp_path)
                
                samples_generated += 1
            
            if samples_generated >= num_samples:
                break
    
    print(f"Generated {samples_generated} prediction visualization overlays")


def main():
    """Main function for standalone CV visualization."""
    parser = argparse.ArgumentParser(description='Generate OpenCV prediction visualizations')
    parser.add_argument('--model-path', default='results/resnet50_model.pth', help='Path to trained model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--samples', type=int, default=12, help='Number of samples to visualize')
    args = parser.parse_args()
    
    print("OpenCV visualization script ready for integration with evaluation pipeline")


if __name__ == "__main__":
    main()
