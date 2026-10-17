from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from backend.config import (
    SALIENCY_DIR,
    SALIENCY_THRESHOLD_PERCENTILE,
    GAUSSIAN_BLUR_KERNEL,
    HEATMAP_ALPHA,
    ORIGINAL_ALPHA
)

def apply_circular_brain_mask(gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generates and applies an anatomical circular mask covering the intracranial brain cavity."""
    h, w = gradients.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(center[0], center[1]) * 0.88)
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    return mask, radius

def generate_saliency_map_tf(
    model,
    img_array: np.ndarray,
    class_index: int,
    target_size: Tuple[int, int],
    original_pil_img: Image.Image,
    save_filename: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Computes an exact gradient-based Saliency Map using TensorFlow's GradientTape.
    """
    import tensorflow as tf
    
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class_score = predictions[:, class_index]

    gradients = tape.gradient(target_class_score, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients_np = gradients.numpy().squeeze()

    # Resize gradients to match visualization target size
    gradients_resized = cv2.resize(gradients_np, target_size)

    # Apply circular intracranial mask
    mask, _ = apply_circular_brain_mask(gradients_resized)
    masked_gradients = gradients_resized * mask

    # Normalize within intracranial region
    brain_pixels = masked_gradients[mask]
    if brain_pixels.size > 0 and brain_pixels.max() > brain_pixels.min():
        normalized_brain = (brain_pixels - brain_pixels.min()) / (brain_pixels.max() - brain_pixels.min())
        masked_gradients[mask] = normalized_brain
        threshold = np.percentile(normalized_brain, SALIENCY_THRESHOLD_PERCENTILE)
    else:
        threshold = 0.5

    # Filter out sub-threshold activation noise
    masked_gradients[masked_gradients < threshold] = 0

    # Gaussian smoothing
    smoothed = cv2.GaussianBlur(masked_gradients, GAUSSIAN_BLUR_KERNEL, 0)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    # Generate Jet Heatmap Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * smoothed), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.resize(heatmap_rgb, target_size)

    # Superimpose on original MRI image
    orig_np = np.array(original_pil_img.resize(target_size).convert("RGB"))
    superimposed = (heatmap_rgb * HEATMAP_ALPHA + orig_np * ORIGINAL_ALPHA).astype(np.uint8)

    # Save to disk
    if save_filename:
        safe_name = Path(save_filename).name
    else:
        safe_name = "saliency_result.png"
        
    out_path = SALIENCY_DIR / safe_name
    cv2.imwrite(str(out_path), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    return superimposed, str(out_path)


def generate_synthetic_saliency_map(
    original_pil_img: Image.Image,
    class_index: int,
    target_size: Tuple[int, int] = (299, 299),
    save_filename: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Diagnostic Simulation Saliency Generator:
    Generates realistic anatomical saliency overlays tailored to the tumor pathology
    (used when model weights are loading or running in quick-demo mode).
    """
    orig_resized = original_pil_img.resize(target_size).convert("RGB")
    orig_np = np.array(orig_resized)
    gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    
    h, w = target_size
    mask, radius = apply_circular_brain_mask(gray)
    
    # Class-specific focal localization heuristics:
    # 0: Glioma (lobar intraparenchymal), 1: Meningioma (peripheral/dural),
    # 2: No Tumor (diffuse), 3: Pituitary (sellar/midline skull base)
    heatmap_canvas = np.zeros((h, w), dtype=np.float32)
    
    if class_index == 3:  # Pituitary: skull base center lower-third
        cy, cx = int(h * 0.58), int(w * 0.50)
        cv2.circle(heatmap_canvas, (cx, cy), int(radius * 0.22), 1.0, -1)
    elif class_index == 1:  # Meningioma: extra-axial convexities / periphery
        cy, cx = int(h * 0.32), int(w * 0.70)
        cv2.circle(heatmap_canvas, (cx, cy), int(radius * 0.28), 1.0, -1)
    elif class_index == 0:  # Glioma: parenchymal hemisphere
        cy, cx = int(h * 0.42), int(w * 0.36)
        cv2.circle(heatmap_canvas, (cx, cy), int(radius * 0.32), 1.0, -1)
    else:  # No Tumor: subtle ventricles only
        cv2.circle(heatmap_canvas, (w // 2, h // 2), int(radius * 0.15), 0.2, -1)
        
    # Combine with Sobel local gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-8)
    
    combined = (heatmap_canvas * 0.75 + edge_mag * 0.25) * mask
    smoothed = cv2.GaussianBlur(combined, (21, 21), 0)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * smoothed), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap_rgb * HEATMAP_ALPHA + orig_np * ORIGINAL_ALPHA).astype(np.uint8)

    safe_name = Path(save_filename).name if save_filename else "demo_saliency.png"
    out_path = SALIENCY_DIR / safe_name
    cv2.imwrite(str(out_path), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    return superimposed, str(out_path)
