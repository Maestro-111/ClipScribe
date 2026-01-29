import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration


class DynamicPrompter:
    def __init__(self, logger, resolver, bert_threshold:float, device="cpu", ignore_text=True):

        self.resolver = resolver

        self.logger = logger
        self.device = device
        self.ignore_text = ignore_text

        self.bert_threshold = bert_threshold

        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

        self.logger.info("Loading BLIP model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        base_stops = self.nlp.Defaults.stop_words

        # DINO doesn't need to know about "picture", "view", or numbers.
        custom_stops = {
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "some",
            "many",
            "several",
            "couple",
            "group",
            "lot",
            "bunch",
            # Possessives & Pronouns
            "his",
            "her",
            "their",
            "my",
            "our",
            "its",
            "another",
            "other",
            # Caption Artifacts
            "image",
            "photo",
            "picture",
            "scene",
            "view",
            "background",
            "foreground",
            "shot",
            "blur",
            "focus",
            "close-up",
            "looking",
            "facing",
        }

        self.stop_words = base_stops.union(custom_stops)

    def clean_chunk(self, chunk):
        """
        Cleans a noun chunk by:
        1. Filtering strict Parts of Speech (removing 'the', 'in', 'on', 'and', '2')
        2. Removing stop words
        3. LEMMATIZING (converting 'cars' -> 'car')
        """

        cleaned_tokens = []

        # POS tags to strictly exclude
        # DET: Determiners (the, a)
        # PRON: Pronouns (he, it)
        # NUM: Numbers (2, 10)
        # PUNCT: Punctuation (., -)
        # CCONJ: Conjunctions (and, or)
        # ADP: Adpositions (in, on, at) - prevents "man in car" -> "man car"

        excluded_pos = {"DET", "PRON", "NUM", "PUNCT", "CCONJ", "ADP", "SPACE"}

        for token in chunk:
            # Check POS and Stop words (using lemma for stop word check is safer)
            if (
                token.pos_ not in excluded_pos
                and token.text.lower() not in self.stop_words
                and token.lemma_.lower() not in self.stop_words
            ):
                # This turns "driving cars" -> "drive car"
                cleaned_tokens.append(token.lemma_)

        if not cleaned_tokens:
            return None

        return " ".join(cleaned_tokens)

    def generate_prompt_from_frame(self, image_rgb, targets: list[str] = None) -> str:

        self.logger.info("blip generating prompt")

        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True).lower()

        self.logger.info(f"Raw Dino Prompt: '{caption}'")

        if not caption or not caption.strip():
            raise ValueError("Generated caption is empty or whitespace only.")

        doc = self.nlp(caption)

        if len(doc) == 0:
            raise ValueError(f"spaCy produced an empty Doc from caption: '{caption}'")

        clean_candidates = set()
        last_valid_cleaned = None  # Track the last valid cleaned chunk

        for chunk in doc.noun_chunks:
            cleaned = self.clean_chunk(chunk)
            if cleaned and len(cleaned) > 2:
                last_valid_cleaned = cleaned  # Update our fallback candidate

                if targets:
                    self.logger.info(f"trying to match {cleaned} with targets array")
                    mapped = self.resolver.resolve(targets, cleaned, threshold=self.bert_threshold)

                    if mapped is not None:
                        clean_candidates.add(mapped)
                else:
                    clean_candidates.add(cleaned)

        if len(clean_candidates) == 0:
            if last_valid_cleaned:
                self.logger.warning(
                    f"Semantic candidates list is empty. Using the last raw label '{last_valid_cleaned}' as fallback.")
                clean_candidates.add(last_valid_cleaned)
            else:
                # If we never even found a valid noun chunk, fallback to the whole caption or a generic term
                self.logger.warning("No valid noun chunks found at all. Using 'object' as extreme fallback.")
                clean_candidates.add("object")

        dino_prompt = " . ".join(list(clean_candidates)) + " ."
        return dino_prompt
