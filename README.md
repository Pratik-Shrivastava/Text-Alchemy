﻿# Text-Alchemy

TEXT-ALCHEMY is an innovative tool designed to help you enhance blurred or distorted text images. By leveraging advanced image processing techniques, TEXT-ALCHEMY produces filtered images with precise details. Once processed, you can easily download the refined image for your use.

## Features

- **Image Enhancement**: Upload images of blurred or distorted text and get a clear, detailed version of the text.
- **Download Option**: After processing, easily download the enhanced image for your records.
- **User-Friendly Interface**: Simple and intuitive UI for seamless user experience.

## How to Use

1. **Upload Image**: Click the 'Upload' button and select the image with blurred or distorted text from your device.
2. **Process Image**: Click the 'Enhance' button to process the uploaded image.
3. **View and Download**: View the enhanced image and click 'Download' to save the image to your device.

## Benefits

- **Students**: Clarify handwritten notes or old documents that are difficult to read.
- **Researchers**: Enhance text in historical documents or manuscripts.
- **Professionals**: Improve readability of scanned documents and reports.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/github-connect.git

2. Navigate to the project directory:
   ```bash
   cd text-alchemy

3. Create virtual environment:
   ```bash
   py -3 -m venv .venv

4. Activate virtual environment:
   ```bash
   .venv\Scripts\activate

5. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the application:
   ```bash
   python main.py

## Contributing

We welcome contributions to GitHub Connect! If you have suggestions for new features, feel free to open an issue or submit a pull request. Please make sure to follow our contribution guidelines.

# Project Revision Notes: Text Image Inpainting via GSDM

## 1. GSDM: Global Structure-guided Diffusion Model

- GSDM is a neural network for text image restoration.
- It leverages the structure of the text as a **prior**, guiding the diffusion model for better restoration.

### GSDM Architecture Stages:
#### I. Structure Prediction Module (SPM)
- Predicts a segmentation map to guide content and positioning of the text.

#### II. Reconstruction Module (RM)
- Takes the predicted segmentation mask and corrupted images as input.
- Generates intact text images with coherent styles.

---

## 2. Handwritten Text OCR Methods

When dealing with **handwritten text images**, two open-source methods are commonly used:

### 2.1 DAN (Decoupled Attention Network)
- Used for Scene Text Recognition (STR).
- Performs well on irregular/curved text using decoupled attention.

**Metrics:**
- **Word Accuracy**: ~82–87%
- **Character Accuracy**: ~91–95%

### 2.2 TrOCR Models (Li et al., 2023)

#### a. TrOCR-L (Large)
- Developed by Microsoft using Transformer + Vision Encoder.
- Fine-tuned for printed and handwritten text.

**Metrics:**
- **Word Accuracy**: ~92–96%
- **Character Accuracy**: ~97–99%

#### b. TrOCR-B (Base)
- Smaller version of TrOCR-L.

**Metrics:**
- **Word Accuracy**: ~88–93%
- **Character Accuracy**: ~95–98%

---

## 3. Image Quality Metrics

Used to evaluate performance of text image restoration:

### 3.1 PSNR (Peak Signal-to-Noise Ratio)
- Measures pixel-wise image quality.
- **Higher is better**
- **Typical range**: 25–35 dB

### 3.2 SSIM (Structural Similarity Index Measure)
- Measures perceptual image quality.
- **Closer to 1.0 is better**
- **Typical range**: 0.80–0.99

---
