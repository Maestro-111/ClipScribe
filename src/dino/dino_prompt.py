import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration


class DynamicPrompter:
    def __init__(self, logger, device="cpu"):
        self.logger = logger
        self.device = device

        self.nlp = spacy.load("en_core_web_sm")

        self.logger.info("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
            self.device)

        self.stop_words = self.nlp.Defaults.stop_words.union({
            "image", "photo", "picture", "scene", "view", "background",
            "foreground", "shot", "close-up", "looking", "facing"
        })

    def clean_chunk(self, chunk):
        """
        Strictly filters chunks to ensure they are physical objects.
        """

        # 1. Head Check: The root of the chunk MUST be a noun.
        if chunk.root.pos_ not in ["NOUN", "PROPN"]:
            return None

        cleaned_tokens = []
        excluded_pos = {"DET", "PRON", "NUM", "PUNCT", "CCONJ", "ADP", "SPACE", "ADV", "VERB"}

        for token in chunk:
            if (token.pos_ not in excluded_pos and
                    token.text.lower() not in self.stop_words and
                    token.lemma_.lower() not in self.stop_words):
                # Lemmatize: 'cars' -> 'car'
                cleaned_tokens.append(token.lemma_)

        if not cleaned_tokens:
            return None

        result = " ".join(cleaned_tokens)
        if len(result) < 2:
            return None

        return result

    def generate_prompt_from_frame(self, image_rgb) -> tuple[str, str]:
        self.logger.info("blip generating prompt")

        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        raw_dino_prompt = self.processor.decode(out[0], skip_special_tokens=True).lower()

        self.logger.info(f"Raw Dino Prompt: '{raw_dino_prompt}'")

        doc = self.nlp(raw_dino_prompt)
        clean_candidates = set()

        for chunk in doc.noun_chunks:
            cleaned = self.clean_chunk(chunk)
            if cleaned:
                clean_candidates.add(cleaned)

        if not clean_candidates:
            self.logger.warning("No valid nouns found. Fallback to 'object'.")
            dino_prompt = "object ."
        else:
            dino_prompt = " . ".join(list(clean_candidates)) + " ."

        return dino_prompt, raw_dino_prompt