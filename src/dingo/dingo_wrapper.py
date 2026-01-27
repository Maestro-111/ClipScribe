import torch
import cv2
from PIL import Image
from dingo.groundingdino.datasets import transforms as T
from dingo.groundingdino.util.inference import load_model, predict
import torchvision.ops as ops



class DingoDetector:
    def __init__(self, config_path="src/dingo/groundingdino/config/GroundingDINO_SwinT_OGC.py", weights_path="checkpoints/groundingdino_swint_ogc.pth"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(config_path, weights_path, device=self.device)

        print(f"Grounding DINO loaded on {self.device}")

    def detect(self, image_cv2, text_prompt, box_threshold=0.35, text_threshold=0.25):

        """
        Input: Raw OpenCV image (BGR)
        Output: List of dicts [{'box': [x1, y1, x2, y2], 'label': 'car', 'score': 0.99}, ...]
        """

        # DINO requires PIL Image in RGB
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        # 2. Inference
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # 3. Post-process coordinates
        # DINO returns normalized [cx, cy, w, h] or [x, y, x, y] depending on version, 
        # We need absolute pixels for SAM 2.
        h, w, _ = image_cv2.shape
        boxes = boxes * torch.Tensor([w, h, w, h])

        boxes_xyxy = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()

        results = []
        for box, score, label in zip(boxes_xyxy, logits, phrases):
            results.append({
                "box": box,  # [x1, y1, x2, y2]
                "label": label,
                "score": float(score)
            })

        return results

    def map_results(self, img, results, output_path, **kwargs):

        for item in results:
            box = item['box']    # [x1, y1, x2, y2]
            label = item['label']
            score = item['score']

            # OpenCV requires integers for coordinates
            x1, y1, x2, y2 = map(int, box)

            # 1. Draw the Box (Green, thickness=2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. Draw the Label (Red text just above the box)
            text = f"{label} ({score:.2f})"
            cv2.putText(
                img,
                text,
                (x1, y1 - 10),            # Position (slightly above top-left corner)
                cv2.FONT_HERSHEY_SIMPLEX, # Font
                0.5,                      # Font scale (size)
                (0, 0, 255),              # Color (Red)
                2                         # Thickness
            )

        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")