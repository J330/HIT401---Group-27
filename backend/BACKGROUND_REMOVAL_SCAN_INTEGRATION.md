# Automatic background removal for the heatmap ONLY

Workflow:
1. DINOv2 similarity gate evaluates the original uploaded photo.
2. The health classifier evaluates the same **original photo** and outputs the class and confidence.
3. Only after classification, `rembg`/`u2netp` isolates the foreground on a neutral grey background and crops it for Grad-CAM. This requires a second model forward pass to explain the **original selected class on the isolated image**. The confidence and diagnosis never come from this second pass.
4. The viewer shows Original / Isolated leaf / Heatmap. The one-page PDF uses the heatmap aligned with the isolated image. There is no user-facing checkbox and the API does not accept a toggle.
5. If removal is unavailable or segmentation fails, Grad-CAM falls back to the original photo. If the first-stage similarity gate fails, neither isolation nor Grad-CAM runs.

**Interpretation:** The isolation step changes the input used to calculate explanation gradients. An isolated-photo Grad-CAM is not the literal attribution of the original-photo prediction, even though it targets the same class. Treat it as a supplementary visual explanation, not evidence of lesion boundaries. Display-only colour enhancement and hotspot expansion remain in place.

Install `pip install -r requirements.txt`. `rembg[cpu]` downloads its U2NetP checkpoint at first use (internet and sufficient cache space required). Check local model performance with your checkpoints. Tests use stubs and don't require those weights.
