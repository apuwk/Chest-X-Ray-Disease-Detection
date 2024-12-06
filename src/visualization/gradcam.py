import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import matplotlib.colors as mcolors
from torchvision import transforms

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Try different target layers in order of preference
        if hasattr(model, 'features'):
            # Get the last convolutional layer
            for module in reversed(model.features):
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
                    break
        elif hasattr(model, 'layer4'):
            self.target_layer = model.layer4[-1]
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
        self.device = next(model.parameters()).device
        
        # Update color scheme to match reference
        self.findings = {
            0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Effusion',
            3: 'Infiltration', 4: 'Mass', 5: 'Nodule',
            6: 'Pneumonia', 7: 'Pneumothorax', 8: 'Consolidation',
            9: 'Edema', 10: 'Emphysema', 11: 'Fibrosis',
            12: 'Pleural_Thickening', 13: 'Hernia'
        }

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_category=None):
        # Get model output
        output = self.model(input_tensor)
        probs = torch.sigmoid(output[0])
        
        if target_category is None:
            # Get category with highest probability
            target_category = torch.argmax(probs).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_category].backward(retain_graph=True)
        
        # Generate CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
        
        # Normalize
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Get all predictions above threshold
        predictions = {self.findings[i]: float(prob) 
                     for i, prob in enumerate(probs) 
                     if float(prob) > 0.01}  # Lower threshold to catch more predictions
        
        return cam, predictions

    def process_image(self, input_image):
        if isinstance(input_image, str):
            image = Image.open(input_image)
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        else:
            image = input_image

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        return input_tensor, np.array(image)

    def create_heatmap(self, cam, original_image):
        # Resize CAM to match original image size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Apply jet colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Convert original image to BGR if needed
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Combine original image with heatmap
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return superimposed

    def process_for_gradio(self, image):
        if image is None:
            return None, None, "No image provided"
        
        # Process image
        input_tensor, original_image = self.process_image(image)
        
        # Generate CAM
        cam, predictions = self.generate_cam(input_tensor)
        
        # Create heatmap
        visualization = self.create_heatmap(cam, original_image)
        
        # Format predictions
        pred_str = "\n".join([
            f"{k}: {v*100:.1f}%" 
            for k, v in sorted(predictions.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        ])
        
        return visualization, pred_str

    def launch_visualization(self):
        # Custom CSS for better styling
        custom_css = """
            #prediction-box { 
                font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
                background: #1a1a1a;
                color: white;
                padding: 20px;
                border-radius: 10px;
            }
            .image-box {
                border-radius: 10px;
                overflow: hidden;
            }
        """
        
        with gr.Blocks(css=custom_css) as demo:
            gr.Markdown("## Chest X-Ray Analysis with GradCAM Visualization")
            
            with gr.Row():
                # Left column for input
                with gr.Column():
                    input_image = gr.Image(
                        type="pil",
                        label="Original Image",
                        elem_classes="image-box"
                    )
                
                # Middle column for visualization
                with gr.Column():
                    output_vis = gr.Image(
                        type="numpy",
                        label="Attention Heatmap",
                        elem_classes="image-box"
                    )
                
                # Right column for predictions
                with gr.Column():
                    predictions = gr.Textbox(
                        label="Predictions",
                        elem_id="prediction-box",
                        lines=10
                    )
            
            analyze_btn = gr.Button("Analyze", variant="primary")
            analyze_btn.click(
                fn=self.process_for_gradio,
                inputs=input_image,
                outputs=[output_vis, predictions]
            )
        
        demo.launch(share=False)