# Seal and Stamp Verification

This project provides a **Seal and Stamp Verification** tool that compares two images to assess their similarity. It uses both **traditional computer vision** and **deep learning techniques** to ensure accurate verification.

## Features

- **Traditional Computer Vision**:
  - Detects and matches features using SIFT (Scale-Invariant Feature Transform).
  - Computes similarity scores based on matched features.

- **Deep Learning**:
  - Leverages ResNet101 for feature extraction.
  - Calculates cosine similarity between feature embeddings.

- **Streamlit Web Interface**:
  - Easy-to-use interface to upload and compare images.
  - Displays visual results with similarity scores and feature match overlays.

## Technologies Used

- **Python**
- **OpenCV**: For image preprocessing and feature matching.
- **TensorFlow**: For deep learning-based feature extraction.
- **Streamlit**: For building an interactive web application.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/Navin-2305-dev/Seal-and-Stamp-Verification.git
