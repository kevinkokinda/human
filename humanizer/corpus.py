import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class CorpusProfile:
    """Lightweight language profile built from a human-written text corpus."""

    def __init__(self, text: str) -> None:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        self._sentences = self._split_sentences(cleaned)
        self._word_counter: Counter[str] = Counter()
        self._bigram_counter: Counter[str] = Counter()
        lead_counter: Counter[str] = Counter()
        tail_counter: Counter[str] = Counter()

        for sentence in self._sentences:
            tokens = self._tokenize(sentence)
            if not tokens:
                continue

            lowered_tokens = [token.lower() for token in tokens]
            self._word_counter.update(lowered_tokens)
            self._bigram_counter.update(self._sliding_phrases(lowered_tokens, 2))

            lead = self._extract_lead(sentence)
            if lead:
                lead_counter[lead] += 1

            tail = self._extract_tail(sentence)
            if tail:
                tail_counter[tail] += 1

        self.common_words: List[str] = [
            word for word, _ in self._word_counter.most_common(600)
        ]
        self.common_bigrams = {
            phrase for phrase, _ in self._bigram_counter.most_common(500)
        }
        self.intro_candidates: List[str] = [
            phrase for phrase, _ in lead_counter.most_common(40)
        ]
        self.outro_candidates: List[str] = [
            phrase for phrase, _ in tail_counter.most_common(40)
        ]
        self.mid_phrases: List[str] = self._derive_mid_phrases()

    @classmethod
    def from_path(cls, path: Path) -> "CorpusProfile":
        text = path.read_text(encoding="utf-8")
        return cls(text)

    def score_text(self, text: str) -> float:
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return 0.0
        lowered = [token.lower() for token in tokens]
        total = len(lowered) - 1
        matches = sum(
            1
            for phrase in self._sliding_phrases(lowered, 2)
            if phrase in self.common_bigrams
        )
        return matches / total if total else 0.0

    def _derive_mid_phrases(self) -> List[str]:
        filler_tokens = {
            "kind",
            "sort",
            "basically",
            "pretty",
            "really",
            "honestly",
            "actually",
            "literally",
            "guess",
            "think",
            "feel",
            "mean",
            "know",
            "just",
        }
        pronouns = {"i", "we", "you", "they"}
        counter: Counter[str] = Counter()
        for phrase, count in self._bigram_counter.items():
            if count < 2:
                continue
            parts = phrase.split()
            if len(parts) != 2:
                continue
            first, second = parts
            if first in pronouns and second in filler_tokens:
                counter[phrase] = count
            elif first in filler_tokens or second in filler_tokens:
                counter[phrase] = count
        return [phrase for phrase, _ in counter.most_common(60)]

    def _split_sentences(self, text: str) -> List[str]:
        sections = _SENTENCE_SPLIT.split(text)
        return [section.strip() for section in sections if section.strip()]

    def _tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text)

    def _sliding_phrases(self, tokens: Sequence[str], size: int) -> Iterable[str]:
        for index in range(len(tokens) - size + 1):
            yield " ".join(tokens[index : index + size])

    def _extract_lead(self, sentence: str) -> str | None:
        if "," not in sentence:
            return None
        head, _ = sentence.split(",", 1)
        cleaned = head.strip()
        word_count = len(cleaned.split())
        if word_count == 0 or word_count > 4:
            return None
        if word_count == 1 and len(cleaned) <= 3:
            return None
        return cleaned.rstrip(",") + ","

    def _extract_tail(self, sentence: str) -> str | None:
        if "," not in sentence:
            return None
        _, tail = sentence.rsplit(",", 1)
        cleaned = tail.strip().rstrip(".!?")
        word_count = len(cleaned.split())
        if word_count < 2 or word_count > 7:
            return None
        if not cleaned or not cleaned[0].islower():
            return None
        return cleaned
