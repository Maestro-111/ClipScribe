import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration


class DynamicPrompter:
    def __init__(self, logger, device="cpu", ignore_text=True):
        self.logger = logger
        self.device = device
        self.ignore_text = ignore_text

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

    def generate_prompt_from_frame(self, image_rgb) -> str:
        """
        generate prompt from frame
        """

        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        self.logger.info(f"Raw Caption: '{caption}'")

        doc = self.nlp(caption.lower())
        clean_candidates = set()

        # Extract Noun Chunks
        for chunk in doc.noun_chunks:
            cleaned = self.clean_chunk(chunk)

            # Filter empty strings and single characters
            if cleaned and len(cleaned) > 2:
                if self.ignore_text and cleaned == "text":
                    continue

                clean_candidates.add(cleaned)

        # DINO Prompt Format
        dino_prompt = " . ".join(list(clean_candidates)) + " ."
        return dino_prompt
