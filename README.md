# Mask R-CNN for Object Segmentation

This repository contains a project demonstrating Mask R-CNN for object detection and instance segmentation. The model predicts both bounding boxes and segmentation masks for objects in an image.

## Why This Project?
Mask R-CNN is widely used in autonomous driving, medical imaging, and smart surveillance, where object segmentation helps in scene understanding, object tracking, and interactive AI applications.

## Dataset Details & Preprocessing
- Uses the COCO dataset, which contains diverse objects with labeled masks.
- Images are resized to match the input shape required by the model.
- Preprocessing includes mean normalization and bounding box adjustments.

## Features
- Uses TensorFlow Object Detection API.
- Implements Mask R-CNN for instance segmentation.
- Capable of detecting and segmenting multiple objects in an image.

## Code Overview
- Loads a pre-trained Mask R-CNN model from TensorFlow Hub.
- Processes input images for inference.
- Runs object detection and segmentation.
- Visualizes results with bounding boxes and masks.
