# GenderStego

GenderStego is a framework for gait-based gender recognition and secure video steganography. The system uses pose estimation and deep learning to classify gender from human gait in videos, then protects privacy by masking people and embedding the original video inside a privacy-preserving skeletal video using steganography and encryption.

## Features

- Pose estimation using MediaPipe to extract human skeletons from video frames
- Gait-based gender classification with a Spatial-Temporal Graph Convolutional Network (ST-GCN)
- Background modeling and blackout masking to remove people from scenes
- Video inpainting to reconstruct backgrounds where people were removed
- Skeletal video generation with color-coded gender skeletons (red for male, blue for female)
- DCT-based steganography to securely embed the original video inside the skeletal video
- AES-256 encryption and HMAC authentication for secure data hiding
- Evaluation metrics: MSE, PSNR, similarity scores, and demographic bias testing

## Requirements

- Python (for pose estimation and deep learning)
- MATLAB (for video processing, inpainting, and steganography)
- MediaPipe, PyTorch or TensorFlow (for ST-GCN model)
- MATLAB Image Processing Toolbox

## Usage

1. Run pose estimation on your input video to extract and save pose data.
2. Train or use the provided ST-GCN model to classify gender from pose sequences.
3. Use MATLAB scripts to generate background, blackout, and inpainted videos.
4. Create skeletal videos by overlaying gender-labeled skeletons.
5. Run the steganography module to embed the original video into the skeletal video.
6. Use the reconstruction module to extract and verify the original video.
7. Check evaluation metrics for quality and fairness.

For detailed instructions, see the documentation in the repository.

## License

This project is for academic and research purposes. See LICENSE file for more details.