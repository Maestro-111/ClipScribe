# Module: Optical Character Recognition

## Purpose
Wraps PaddleOCR to extract text from video frames and implements logic to group raw text fragments into logical lines and blocks.

## Key Files
* `paddle_wrapper.py`: Initializes PaddleOCR. Contains crucial post-processing clustering logic (`_consolidate_boxes_hierarchical`, `_consolidate_boxes_dbscan`) to merge nearby bounding boxes into coherent sentences rather than fragmented words.

## Guidelines
* If text detection is too fragmented or combining unrelated lines, adjust the `y_thresh` and `x_thresh` clustering logic in this wrapper.
