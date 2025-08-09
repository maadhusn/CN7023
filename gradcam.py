"""Grad-CAM implementation for CNN explainability."""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse


class GradCAM:
    """Grad-CAM implementation for CNN visualization."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap."""
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_score = output[:, class_idx].squeeze()
        class_score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=[2, 3])
        
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    """Apply colormap on image."""
    activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * 0.4 + org_im * 0.6
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    return superimposed_img


def generate_gradcam_visualizations(model, test_loader, class_names, device, num_samples=24, save_dir='results'):
    """Generate Grad-CAM visualizations for correct and incorrect predictions."""
    model.eval()
    
    target_layer = model.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)
    
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for i in range(data.size(0)):
                sample_info = {
                    'image': data[i:i+1],
                    'true_label': target[i].item(),
                    'pred_label': pred[i].item(),
                    'confidence': F.softmax(output[i], dim=0)[pred[i]].item(),
                    'batch_idx': batch_idx,
                    'sample_idx': i
                }
                
                if pred[i] == target[i]:
                    correct_samples.append(sample_info)
                else:
                    incorrect_samples.append(sample_info)
                
                if len(correct_samples) + len(incorrect_samples) >= num_samples * 2:
                    break
            
            if len(correct_samples) + len(incorrect_samples) >= num_samples * 2:
                break
    
    for i, sample in enumerate(correct_samples[:num_samples//2]):
        cam = gradcam.generate_cam(sample['image'])
        
        img_tensor = sample['image'].squeeze().cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        gradcam_img = apply_colormap_on_image(img_np, cam)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(img_np)
        ax1.set_title(f'Original\nTrue: {class_names[sample["true_label"]]}')
        ax1.axis('off')
        
        ax2.imshow(gradcam_img)
        ax2.set_title(f'Grad-CAM\nPred: {class_names[sample["pred_label"]]} ({sample["confidence"]:.2f})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gradcam_correct_{i+1:02d}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    for i, sample in enumerate(incorrect_samples[:num_samples//2]):
        cam = gradcam.generate_cam(sample['image'])
        
        img_tensor = sample['image'].squeeze().cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        gradcam_img = apply_colormap_on_image(img_np, cam)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(img_np)
        ax1.set_title(f'Original\nTrue: {class_names[sample["true_label"]]}')
        ax1.axis('off')
        
        ax2.imshow(gradcam_img)
        ax2.set_title(f'Grad-CAM\nPred: {class_names[sample["pred_label"]]} ({sample["confidence"]:.2f})')
        ax2.axis('off')
        
        plt.tight_layout()
        
        true_name = class_names[sample["true_label"]].replace(' ', '_')
        pred_name = class_names[sample["pred_label"]].replace(' ', '_')
        filename = f'gradcam_missed_{i+1:02d}__{pred_name}_vs_{true_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Generated {len(correct_samples[:num_samples//2])} correct and {len(incorrect_samples[:num_samples//2])} incorrect Grad-CAM visualizations")


def main():
    """Main function for standalone Grad-CAM generation."""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model-path', default='results/resnet50_model.pth', help='Path to trained model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--samples', type=int, default=24, help='Number of samples to visualize')
    args = parser.parse_args()
    
    print("Grad-CAM visualization script ready for integration with evaluation pipeline")


if __name__ == "__main__":
    main()
