import torch
import cv2
from PIL import Image
from dino.groundingdino.datasets import transforms as T
from dino.groundingdino.util.inference import load_model, predict
import torchvision.ops as ops
import os

class DinoDetector:
    def __init__(
        self,
        logger,
        dino_type="tiny",
        weights_dir="checkpoints",
        device="cpu",
    ):
        self.logger = logger
        self.device = device

        if dino_type == "tiny":
            config_path = "src/dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_path = "groundingdino_swint_ogc.pth"
        elif dino_type == "base":
            config_path = "src/dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
            weights_path = "groundingdino_swinb_cogcoor.pth"
        else:
            raise ValueError("dino_type must be 'tiny' or 'base'")

        self.logger.info(f"configured paths for {dino_type} dino")
        self.logger.info(f"config_path {config_path}; weights_path {weights_path}")

        self.model = load_model(config_path, os.path.join(weights_dir, weights_path), device=self.device)
        self.logger.info(f"Grounding DINO loaded on {self.device}")

    def detect(self, image_cv2, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Input: Raw OpenCV image (BGR)
        Output: List of dicts [{'box': [x1, y1, x2, y2], 'label': 'car', 'score': 0.99}, ...]
        """

        self.logger.info("dino detection")

        # DINO requires PIL Image in RGB
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # DINO returns normalized [cx, cy, w, h] or [x, y, x, y] depending on version,
        # need absolute pixels for SAM 2.
        h, w, _ = image_cv2.shape
        boxes = boxes * torch.Tensor([w, h, w, h])

        boxes_xyxy = (
            ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            .cpu()
            .numpy()
            .tolist()
        )

        results = []
        for box, score, label in zip(boxes_xyxy, logits, phrases):
            results.append(
                {"box": box, "label": label, "score": float(score)}  # [x1, y1, x2, y2]
            )

        return results

    def map_results(self, img, results, output_path, **kwargs):
        vis_img = img.copy()

        for item in results:
            box = item["box"]  # [x1, y1, x2, y2]
            label = item["label"]
            score = item["score"]

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} ({score:.2f})"
            font_scale = 0.5
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            text_x = x1
            text_y = y1 - 5

            if y1 < 20:
                text_y = y1 + text_height + 5

            if x1 + text_width > vis_img.shape[1]:
                text_x = vis_img.shape[1] - text_width - 5

            cv2.rectangle(
                vis_img,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                (0, 0, 0),
                cv2.FILLED,
            )

            cv2.putText(
                vis_img,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

        cv2.imwrite(output_path, vis_img)
        self.logger.info(f"Saved visualization to {output_path}")
