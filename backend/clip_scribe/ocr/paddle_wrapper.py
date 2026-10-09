import logging
import numpy as np
from paddleocr import PaddleOCR
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import fclusterdata

logging.getLogger("ppocr").setLevel(logging.WARNING)  # suprres ocr logs?
logger = logging.getLogger("clip_scribe")


class OCRSystem:
    @staticmethod
    def visualize_results(image, results, show_text=True, show_confidence=True):
        """
        Draw OCR results on the image.

        Args:
            image: Input image (numpy array, BGR or RGB)
            results: List of OCR results from detect()
            show_text: Whether to draw the recognized text
            show_confidence: Whether to show confidence scores

        Returns:
            Annotated image (numpy array)
        """
        import cv2

        # Make a copy to avoid modifying the original
        vis_img = image.copy()

        # If image is grayscale, convert to BGR for colored annotations
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        for result in results:
            box = result["box"]
            text = result["text"]
            confidence = result["confidence"]

            # Unpack coordinates
            x1, y1, x2, y2 = box

            # Choose color based on confidence (green=high, yellow=medium, red=low)
            if confidence > 0.9:
                color = (0, 255, 0)  # Green
            elif confidence > 0.7:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label_parts = []
            if show_text:
                # Truncate long text
                display_text = text[:30] + "..." if len(text) > 30 else text
                label_parts.append(display_text)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")

            label = " | ".join(label_parts)

            if label:
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    vis_img,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 5, y1),
                    color,
                    -1,  # Filled
                )

                # Draw text
                cv2.putText(
                    vis_img,
                    label,
                    (x1 + 2, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    cv2.LINE_AA,
                )

        return vis_img

    @staticmethod
    def _consolidate_boxes_simple(raw_results, x_thresh=50, y_thresh=20):
        """
        Original greedy merge approach for comparison.
        """

        if not raw_results:
            return []

        sorted_boxes = sorted(raw_results, key=lambda b: (b["box"][1], b["box"][0]))

        merged = []
        current_group = sorted_boxes[0].copy()

        for next_box in sorted_boxes[1:]:
            c_box = current_group["box"]
            n_box = next_box["box"]

            c_center_y = (c_box[1] + c_box[3]) / 2
            n_center_y = (n_box[1] + n_box[3]) / 2
            vertical_diff = abs(c_center_y - n_center_y)
            horizontal_gap = n_box[0] - c_box[2]

            if vertical_diff < y_thresh and horizontal_gap < x_thresh:
                new_x1 = min(c_box[0], n_box[0])
                new_y1 = min(c_box[1], n_box[1])
                new_x2 = max(c_box[2], n_box[2])
                new_y2 = max(c_box[3], n_box[3])

                current_group = {
                    "box": [new_x1, new_y1, new_x2, new_y2],
                    "text": current_group["text"] + " " + next_box["text"],
                    "label": "text_group",
                    "confidence": (current_group["confidence"] + next_box["confidence"])
                    / 2,
                }
            else:
                merged.append(current_group)
                current_group = next_box.copy()

        merged.append(current_group)
        return merged

    @staticmethod
    def _merge_box_group(boxes):
        """
        Merge a group of boxes into a single box with concatenated text.

        Args:
            boxes: List of box dicts to merge

        Returns:
            Single merged box dict
        """
        if len(boxes) == 1:
            return boxes[0].copy()

        # Calculate bounding box
        x_min = min(b["box"][0] for b in boxes)
        y_min = min(b["box"][1] for b in boxes)
        x_max = max(b["box"][2] for b in boxes)
        y_max = max(b["box"][3] for b in boxes)

        # Concatenate text (already sorted left-to-right)
        merged_text = " ".join(b["text"] for b in boxes)

        # Average confidence
        avg_confidence = sum(b["confidence"] for b in boxes) / len(boxes)

        return {
            "box": [x_min, y_min, x_max, y_max],
            "text": merged_text,
            "label": "text_group",
            "confidence": avg_confidence,
        }

    def __init__(
        self,
        lang="en",
        use_textline_orientation=True,
        merge_lines=True,
        merge_method="hierarchical",
    ):
        """
        Initialize PaddleOCR 3.x with correct parameters.

        Args:
            lang: Language code (default: 'en')
            use_textline_orientation: Whether to classify text orientation
            merge_lines: Whether to merge adjacent text boxes into lines
            merge_method: 'hierarchical', 'dbscan', or 'simple' (the old greedy method)
        """
        logger.info(f"Loading PaddleOCR (lang={lang})...")

        self.ocr_engine = PaddleOCR(
            use_textline_orientation=use_textline_orientation,
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

        self.merge_lines = merge_lines
        self.merge_method = merge_method
        logger.info("PaddleOCR loaded.")

    def detect(self, image_cv2):
        """
        Run OCR on an image and return structured results.

        Args:
            image_cv2: OpenCV image (numpy array)

        Returns:
            List of dicts with keys: box, text, label, confidence
        """
        # 1. Use the native .ocr() method instead of .predict()
        raw_results = self.ocr_engine.ocr(image_cv2)
        parsed_results = []

        # PaddleOCR returns a list of lists (one list per image).
        # Check if we got valid results back for our single image.
        if raw_results and raw_results[0]:
            for line in raw_results[0]:
                if not line:
                    continue

                # Standard 'line' format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('text', confidence)]
                box_points, (text, score) = line

                # Extract coordinates to build the axis-aligned [x_min, y_min, x_max, y_max] box
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                parsed_results.append(
                    {
                        "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "text": text,
                        "label": "text",
                        "confidence": float(score),
                    }
                )

        # 2. Merge lines logic remains identical
        if self.merge_lines and len(parsed_results) > 0:
            if self.merge_method == "hierarchical":
                return self._consolidate_boxes_hierarchical(parsed_results)
            elif self.merge_method == "dbscan":
                return self._consolidate_boxes_dbscan(parsed_results)
            else:
                return self._consolidate_boxes_simple(parsed_results)

        return parsed_results

    def _consolidate_boxes_hierarchical(self, raw_results, y_thresh=20):
        """
        Merge text boxes using hierarchical clustering on Y-coordinates.
        This works well for text lines which are primarily horizontal.

        Args:
            raw_results: List of detection results
            y_thresh: Maximum vertical distance to consider boxes on same line

        Returns:
            List of merged detection results
        """
        if not raw_results:
            return []

        if len(raw_results) == 1:
            return raw_results

        # Extract center Y coordinates for clustering
        centers_y = np.array(
            [(box["box"][1] + box["box"][3]) / 2.0 for box in raw_results]
        ).reshape(-1, 1)

        # Use hierarchical clustering to group boxes by Y-coordinate
        # This automatically finds the number of lines
        labels = fclusterdata(
            centers_y,
            t=y_thresh,  # Distance threshold
            criterion="distance",
            method="complete",
        )

        # Group boxes by cluster label
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(raw_results[i])

        # Merge boxes within each cluster (line)
        merged = []
        for cluster_boxes in clusters.values():
            # Sort left-to-right within the line
            cluster_boxes.sort(key=lambda b: b["box"][0])

            # Merge all boxes in this line
            merged_box = self._merge_box_group(cluster_boxes)
            merged.append(merged_box)

        # Sort lines top-to-bottom
        merged.sort(key=lambda b: b["box"][1])

        return merged

    def _consolidate_boxes_dbscan(self, raw_results, eps=30, min_samples=1):
        """
        Merge text boxes using DBSCAN clustering on 2D centers.
        More flexible than hierarchical, can handle non-horizontal text.

        Args:
            raw_results: List of detection results
            eps: Maximum distance between boxes to be in same cluster
            min_samples: Minimum boxes to form a cluster (1 = allow single-box clusters)

        Returns:
            List of merged detection results
        """
        if not raw_results:
            return []

        if len(raw_results) == 1:
            return raw_results

        # Extract box centers for clustering
        centers = np.array(
            [
                [
                    (box["box"][0] + box["box"][2]) / 2.0,
                    (box["box"][1] + box["box"][3]) / 2.0,
                ]
                for box in raw_results
            ]
        )

        # Weight Y-coordinate more heavily (text lines are horizontal)
        centers_weighted = centers.copy()
        centers_weighted[:, 1] *= 2  # Double the weight of Y-coordinate

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_weighted)
        labels = clustering.labels_

        # Group boxes by cluster label
        clusters = {}
        for i, label in enumerate(labels):
            # label == -1 means noise/outlier in DBSCAN
            # We'll keep these as individual boxes
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(raw_results[i])

        # Merge boxes within each cluster
        merged = []
        for cluster_boxes in clusters.values():
            # Sort left-to-right
            cluster_boxes.sort(key=lambda b: b["box"][0])

            # Merge all boxes in this cluster
            merged_box = self._merge_box_group(cluster_boxes)
            merged.append(merged_box)

        # Sort top-to-bottom
        merged.sort(key=lambda b: b["box"][1])

        return merged

    def save_visualization(self, image, results, output_path, **kwargs):
        """
        Save visualization to file.

        Args:
            image: Input image
            results: OCR results
            output_path: Path to save the image
            **kwargs: Additional arguments for visualize_results()
        """
        import cv2

        vis_img = self.visualize_results(image, results, **kwargs)
        cv2.imwrite(output_path, vis_img)
        logger.info(f"OCR Visualization saved to: {output_path}")
