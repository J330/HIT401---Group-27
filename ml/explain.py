# ml/explain.py
# Grad-CAM for CNNs (EfficientNetV2-S, ConvNeXt V2 Base) + the reshape needed
# to run the same technique on transformers (Swin, ViT, DINOv2).
# Source: https://github.com/jacobgil/pytorch-grad-cam
#         https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def explain_cnn(model, target_layers, input_tensor, predicted_class):
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])


def reshape_transform(tensor, height=14, width=14):
    # Converts a transformer's flat patch tokens back into a spatial grid so
    # Grad-CAM can draw a heatmap over it, same as it does for a CNN.
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def explain_transformer(model, target_layer, input_tensor, predicted_class):
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    return cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])
