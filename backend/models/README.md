# Model Artifacts Directory

Store your trained deep learning weight files here:

- `xception_model.weights.h5`: Weights for the Transfer Learning Xception architecture (input shape: 299x299x3).
- `cnn_model.h5` or `cnn_model.keras`: Full model for the 4-layer Custom CNN architecture (input shape: 224x224x3).

### How were these models trained?
Refer to [`SIC_Capstone_Project (3).ipynb`](../../SIC_Capstone_Project%20(3).ipynb) for full training logs, loss curves, confusion matrices, and metrics.
When running in Colab, the notebook exports:
1. `model.save_weights("xception_model.weights.h5")`
2. `cnn_model.save("cnn_model.h5")`

> **Note:** If trained weights are not present, Tumor Vision automatically activates its smart **Diagnostic Demo Mode**, allowing you to explore the full UI, saliency visualization, and Gemini neurologist explainer with high-fidelity baseline representations.
