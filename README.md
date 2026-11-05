# MaskifyAI - Mask R-CNN Instance Segmentation

MaskifyAI demonstrates **object detection and instance segmentation using a pre-trained Mask R-CNN model from TensorFlow Hub**.

The project uses **Mask R-CNN with an Inception-ResNet-V2 backbone** to identify objects in an image, generate bounding boxes, assign class labels and confidence scores and produce a separate segmentation mask for each detected object.

## What the Project Does

The notebook performs the following steps:

1. Installs the TensorFlow Object Detection API.
2. Loads test images from local paths or image URLs.
3. Loads a pre-trained Mask R-CNN model from TensorFlow Hub.
4. Runs inference on the selected image.
5. Extracts object detection and segmentation outputs.
6. Converts predicted masks from bounding-box coordinates to full-image masks.
7. Applies a threshold to the segmentation masks.
8. Visualizes detected objects using bounding boxes, class labels, confidence scores and instance masks.

## Model

The project uses:

**Mask R-CNN Inception ResNet V2 1024×1024**

TensorFlow Hub model:

```text
https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1
```

Mask R-CNN extends object detection by predicting a segmentation mask for each detected object in addition to its bounding box and class.

## Image Input

The notebook includes several example images:

- Beach
- Dogs
- Phones
- Street

Images can be loaded either from local files or directly from URLs.

The selected image is converted into a NumPy array with the shape:

```text
(1, height, width, 3)
```

before being passed to the model.

## Instance Segmentation Pipeline

After inference, the model returns outputs including:

- Detection boxes
- Detection classes
- Detection scores
- Detection masks

The predicted masks initially correspond to the detected bounding boxes.

The notebook uses TensorFlow Object Detection API utilities to transform these masks into full-image coordinates:

```python
utils_ops.reframe_box_masks_to_image_masks(...)
```

The resulting masks are thresholded at:

```text
0.6
```

before visualization.

## Visualization

The project uses the **MS COCO label map** from the TensorFlow Object Detection API to convert predicted class IDs into object names.

The final visualization includes:

- Bounding boxes
- Object class labels
- Confidence scores
- Instance segmentation masks

Only detections with a confidence score of at least:

```text
0.70
```

are displayed.

## Technologies Used

- Python
- TensorFlow
- TensorFlow Hub
- TensorFlow Object Detection API
- Mask R-CNN
- Inception-ResNet-V2
- NumPy
- PIL
- Matplotlib
- MS COCO label map

## Key Concepts Demonstrated

- Object detection
- Instance segmentation
- Pre-trained deep learning models
- TensorFlow Hub model inference
- Bounding-box prediction
- Segmentation-mask processing
- Confidence-based detection filtering
- Computer vision visualization

## Possible Extensions

The project could be extended by:

- Testing the model on custom datasets
- Comparing results across different images
- Evaluating segmentation performance quantitatively
- Fine-tuning an instance-segmentation model for a specific application
- Comparing Mask R-CNN with newer segmentation architectures
