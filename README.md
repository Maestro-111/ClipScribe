# ClipScribe

A video processing toolkit for object detection, tracking, and OCR extraction using state-of-the-art AI models.

## Overview

ClipScribe combines GroundingDINO for object detection, SAM2 (Segment Anything Model 2) for segmentation and tracking, and PaddleOCR for text extraction from video content. This tool is designed to extract visual and textual information from videos, making it ideal for video analysis, content extraction, and automated annotation tasks.

## Features

- **Object Detection**: Leverages GroundingDINO for text-prompted object detection
- **Video Segmentation & Tracking**: Uses SAM2.1 for precise object segmentation and temporal tracking
- **OCR Extraction**: Integrates PaddleOCR for text recognition in video frames
- **Batch Processing**: Process multiple videos efficiently
- **Artifact Generation**: Saves detection results, OCR outputs, and tracked videos

## Third-Party Code

This project includes third-party components:

- SAM2 by Meta Platforms, Inc. (Apache License 2.0) [SAM2](https://github.com/facebookresearch/segment-anything-2)
- GroundingDINO (Apache License 2.0 / MIT) [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

Their respective licenses are included in the source tree.
