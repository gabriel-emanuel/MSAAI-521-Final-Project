# Skin Cancer Classification and Detection

![Skin Cancer Detection] <!-- Replace with an actual image or remove this line if not applicable -->

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction

Skin cancer is one of the most prevalent and rapidly increasing forms of cancer worldwide. Early detection and accurate classification are critical for effective treatment and improved patient outcomes. Traditional diagnostic methods rely heavily on the expertise of dermatologists to interpret dermoscopic images, which can be time-consuming and subject to human error.

Our project explores and compares various deep learning-based computer vision models for the classification and detection of skin cancer. Utilizing the HAM10000 dataset, we evaluate transformer-based Vision Transformers (ViT) and convolutional neural network (CNN) architectures, specifically DenseNet variants (DenseNet-121, DenseNet-169, and DenseNet-201). Additionally, we incorporate object detection using the YOLOv7 model to localize lesions within images.

The ultimate goal is to develop a practical mobile application that serves as a preliminary screening tool for the general public, encouraging timely medical consultations.

## Features

- **Classification Models:**
  - Vision Transformer (ViT)
  - DenseNet-121, DenseNet-169, DenseNet-201
  - Models trained with and without data augmentation

- **Object Detection:**
  - YOLOv7 for lesion localization

- **Data Handling:**
  - Comprehensive data preprocessing and augmentation to address class imbalance

- **Evaluation:**
  - Detailed performance metrics including accuracy, precision, recall, and F1-score
  - Learning curves analysis for overfitting assessment

- **Mobile Application Prototype:**
  - Interactive Gradio-based demo for real-time classification and detection

## Technologies Used

- **Programming Languages:** Python
- **Libraries & Frameworks:**
  - TensorFlow & Keras
  - PyTorch
  - Hugging Face Transformers
  - OpenCV
  - NumPy
  - PIL (Pillow)
  - scikit-learn
- **Tools:**
  - Google Colab
  - Git & GitHub

## Dataset

**HAM10000 Dataset**

- **Description:** A large collection of multi-source dermatoscopic images of common pigmented skin lesions.
- **Number of Images:** 10,015
- **Classes:**
  - akiec: Actinic keratoses and intraepithelial carcinoma
  - bcc: Basal cell carcinoma
  - bkl: Benign keratosis-like lesions
  - df: Dermatofibroma
  - mel: Melanoma
  - nv: Melanocytic nevi
  - vasc: Vascular lesions

**Data Source:**  
[Tschandl et al., 2018](https://doi.org/10.1038/sdata.2018.161)

## Models

### 1. Vision Transformer (ViT)

- **Architecture:** Transformer-based model pretrained on ImageNet.
- **Parameters:** 85 Million
- **Performance:**
  - Accuracy: 92%
  - Precision: 0.79
  - Recall: 0.71
  - F1-Score: 0.74

### 2. DenseNet Variants

- **Architectures:** DenseNet-121, DenseNet-169, DenseNet-201
- **Parameters:** 
  - DenseNet-121: 8 Million
  - DenseNet-169: 14 Million
  - DenseNet-201: 20 Million
- **Performance with Data Augmentation:**

| **Model**          | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|--------------------|--------------|---------------|------------|--------------|
| DenseNet-121 (v3)  | 0.86         | 0.60          | 0.55       | 0.57         |
| DenseNet-169 (v3)  | 0.87         | 0.65          | 0.61       | 0.63         |
| DenseNet-201 (v3)  | 0.87         | 0.61          | 0.56       | 0.57         |

**Key Insights:**

- Data augmentation significantly improved accuracy and precision across all DenseNet models.
- DenseNet-169 achieved the highest F1-score, balancing precision and recall effectively.

### 3. Object Detection: YOLOv7

- **Purpose:** Localize skin cancer lesions within images.
- **Performance:** Accurate bounding boxes around lesions; effective for preliminary screening.
- **Implementation:** Trained on segmentation masks from HAM10000.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/gabriel-emanuel/MSAAI-521-Final-Project.git
cd skin-cancer-detector
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*Ensure that your `requirements.txt` includes all necessary libraries:*

```plaintext
tensorflow
torch
transformers
gradio
opencv-python
numpy
pillow
scikit-learn
matplotlib
seaborn
```

## Usage

### Running the Classification Models

Navigate to the `models` directory and run the evaluation scripts.

```bash
python Evaluate_Classifiers.py
```

This script will evaluate all DenseNet and ViT models on the validation set, generating performance metrics and learning curves.

### Object Detection with YOLOv7

Ensure that the YOLOv7 model is properly trained and the weights are available.

```python
# Example usage in Python
from detection import detect_and_draw_boxes
from PIL import Image

image = Image.open('path_to_image.jpg')
detected_image = detect_and_draw_boxes(image)
detected_image.show()
```

### Launching the Gradio Demo

Run the Gradio application to interact with the classification and detection models.

```bash
python app.py
```

Access the demo via the provided local URL or the public link if deployed online.

## Evaluation

**Performance Metrics:**

- **Vision Transformer (ViT):**
  - Accuracy: 92%
  - Precision: 0.79
  - Recall: 0.71
  - F1-Score: 0.74

- **DenseNet-169 with Augmentation:**
  - Accuracy: 87%
  - Precision: 0.65
  - Recall: 0.61
  - F1-Score: 0.63

**Findings:**

- ViT outperforms DenseNet variants in all classification metrics.
- Data augmentation enhances precision and overall accuracy in DenseNet models.
- YOLOv7 effectively localizes lesions, supporting preliminary screening efforts.

## Future Work

- **Data Augmentation Enhancements:**
  - Implement advanced techniques like GANs for synthetic data generation.
  - Expand dataset with more samples for underrepresented classes.

- **Model Optimization:**
  - Optimize ViT for mobile deployment through techniques like quantization and pruning.
  - Explore ensemble methods combining ViT and DenseNet strengths.

- **Object Detection Refinement:**
  - Improve YOLOv7's precision and reduce false positives.
  - Integrate real-time video feed for dynamic lesion detection.

- **Clinical Validation:**
  - Collaborate with dermatologists to validate model predictions on external datasets.
  - Ensure compliance with medical standards and regulations.

- **Mobile Application Development:**
  - Develop a user-friendly interface for the mobile app.
  - Incorporate features like lesion tracking over time and user feedback mechanisms.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

**Group 5 Members:**

- **Gabriel Emanuel Colón**  
  Email: gabriel.colon@usd.edu

- **Geoffrey Fadera**  
  Email: geoffrey.fadera@usd.edu

- **Yunus Tezcan**  
  Email: yunus.tezcan@usd.edu

**University of San Diego, MSAAI**

## Acknowledgements

- **HAM10000 Dataset:** [Tschandl et al., 2018](https://doi.org/10.1038/sdata.2018.161)
- **YOLOv7 Model:** [Ultralytics](https://github.com/ultralytics/YOLOv7)
- **Vision Transformer:** [Hugging Face Transformers](https://huggingface.co/google/vit-base-patch16-224)
- **Gradio:** [Gradio Documentation](https://gradio.app/get_started)
- **Research Inspirations:**
  - [Ćirković & Stanić, 2024](https://scidar.kg.ac.rs/handle/123456789/21167)
  - [Gloster & Brodland, 1996](https://doi.org/10.1111/j.1524-4725.1996.tb00312.x)
  - [Tschandl et al., 2020](https://doi.org/10.1038/s41591-020-0942-0)
