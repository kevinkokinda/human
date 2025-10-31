import random
import re
from typing import Iterable, List, Tuple

from .corpus import CorpusProfile


class Humanizer:
    _phrase_variants = [
        ("in order to", ["to", "so I can"]),
        ("completion of the", ["finishing the", "wrapping up the"]),
        ("optimization of the", ["optimizing the", "dialing in the"]),
        ("for example", ["for instance", "say for example"]),
        ("for instance", ["like", "just as an example"]),
        ("in addition", ["on top of that", "plus"]),
        ("additionally", ["on top of that", "besides"]),
        ("furthermore", ["plus", "on top of that"]),
        ("moreover", ["and on top of that", "plus"]),
        ("however", ["even so", "still"]),
        ("therefore", ["so", "because of that"]),
        ("thus", ["so", "because of that"]),
        ("consequently", ["so", "as a result"]),
        ("overall", ["all in all", "taken together"]),
        ("in conclusion", ["to wrap things up", "when I pull it together"]),
        ("importantly", ["the big thing is", "most of all"]),
        ("in summary", ["to sum it up", "put simply"]),
        ("in short", ["long story short", "short version"]),
        ("as a result", ["because of that", "so that meant"]),
        ("is necessary for", ["is needed for", "is basically required for"]),
        ("prior to", ["before", "right before"]),
        ("assist", ["help", "back up"]),
    ]

    _word_variants = [
        ("utilize", ["use", "lean on"]),
        ("significant", ["big", "notable"]),
        ("impact", ["effect", "shift"]),
        ("challenge", ["hurdle", "rough patch"]),
        ("issue", ["problem", "snag"]),
        ("resolve", ["fix", "sort out"]),
        ("approach", ["way", "tactic"]),
        ("objective", ["goal", "aim"]),
        ("methodology", ["approach", "way we tackled it"]),
        ("advantage", ["plus", "upside"]),
        ("disadvantage", ["downside", "trade-off"]),
        ("consider", ["think about", "weigh"]),
        ("necessary", ["needed", "pretty essential"]),
        ("difficult", ["tough", "no cakewalk"]),
        ("straightforward", ["simple", "pretty direct"]),
        ("complex", ["complicated", "layered"]),
        ("numerous", ["a bunch of", "plenty of"]),
        ("require", ["need", "call for"]),
        ("leverage", ["use", "lean on"]),
        ("maximize", ["make the most of", "squeeze more out of"]),
        ("minimize", ["cut down", "keep small"]),
        ("optimize", ["fine-tune", "dial in"]),
        ("achieve", ["hit", "pull off"]),
    ]

    _contractions = [
        ("do not", "don't"),
        ("does not", "doesn't"),
        ("did not", "didn't"),
        ("can not", "can't"),
        ("cannot", "can't"),
        ("will not", "won't"),
        ("would not", "wouldn't"),
        ("should not", "shouldn't"),
        ("could not", "couldn't"),
        ("are not", "aren't"),
        ("is not", "isn't"),
        ("were not", "weren't"),
        ("was not", "wasn't"),
        ("have not", "haven't"),
        ("has not", "hasn't"),
        ("had not", "hadn't"),
        ("i am", "I'm"),
        ("we are", "we're"),
        ("they are", "they're"),
        ("you are", "you're"),
        ("it is", "it's"),
        ("that is", "that's"),
        ("there is", "there's"),
        ("here is", "here's"),
        ("let us", "let's"),
    ]

    _formal_leads = {
        "however": ["even so", "still", "that said"],
        "therefore": ["so", "because of that"],
        "thus": ["so", "because of that"],
        "additionally": ["on top of that", "plus"],
        "furthermore": ["plus", "and on top of that"],
        "moreover": ["plus", "beyond that"],
        "consequently": ["so", "as a result"],
        "nevertheless": ["even then", "still"],
        "nonetheless": ["still", "even then"],
    }

    _intro_leans = [
        "Honestly,",
        "From my side,",
        "If I'm being real,",
        "To be fair,",
        "Practically speaking,",
        "For what it's worth,",
        "Between us,",
        "Quick reality check,",
    ]

    _outro_tails = [
        "at least from what I've seen",
        "from where I'm sitting",
        "based on how this played out",
        "in the day-to-day reality",
        "and that's been pretty true here",
        "which lines up with my experience",
    ]

    _hedge_tokens = [
        "kind of",
        "pretty much",
        "basically",
        "sort of",
        "mostly",
        "generally",
    ]

    _emphasis_tokens = [
        "really",
        "genuinely",
        "actually",
        "personally",
        "frankly",
    ]

    _pair_merge_starters = {"I", "It", "This", "That", "We", "You", "They"}

    _intro_blockers = [
        "still",
        "even so",
        "even then",
        "that said",
        "so",
        "plus",
        "what's more",
        "and",
        "but",
        "because of that",
        "as a result",
        "to wrap things up",
        "when i pull it together",
    ]

    def __init__(
        self,
        creativity: float = 0.65,
        seed: int | None = None,
        corpus: CorpusProfile | None = None,
    ) -> None:
        self.creativity = max(0.0, min(1.0, creativity))
        self.rng = random.Random(seed)
        self.corpus_profile = corpus

        self.intro_leans = list(self._intro_leans)
        self.outro_tails = list(self._outro_tails)
        self.hedge_tokens = list(self._hedge_tokens)
        self.emphasis_tokens = list(self._emphasis_tokens)
        self.casual_need_to = ["really need to", "kind of have to", "definitely need to", "have to"]
        self.casual_need_plain = ["really need", "kind of need", "definitely need", "actually need"]
        self.casual_have_variants = ["We've got", "We have", "We basically have"]
        self.casual_still_have_variants = ["We still have", "We've still got", "We basically still have"]
        self.allows_we_variants = ["That lets us", "That means we can", "That gives us room to", "That basically lets us"]
        self.allows_me_variants = ["That lets me", "That means I can", "That gives me room to"]
        self.allows_generic_variants = ["That lets {group}", "That means {group} can", "That gives {group} room to", "That basically lets {group}"]
        self.pair_merge_starters = set(self._pair_merge_starters)
        self.intro_blockers = list(self._intro_blockers)
        self.corpus_mid_phrases: List[str] = []

        if self.corpus_profile:
            self.intro_leans = self._merge_unique(self.intro_leans, self.corpus_profile.intro_candidates)
            self.outro_tails = self._merge_unique(self.outro_tails, self.corpus_profile.outro_candidates)
            self.corpus_mid_phrases = list(self.corpus_profile.mid_phrases)
            hedge_candidates = [
                phrase
                for phrase in self.corpus_mid_phrases
                if 1 < len(phrase.split()) <= 3 and phrase.split()[0] not in {"i", "we", "you", "they"}
            ]
            self.hedge_tokens = self._merge_unique(self.hedge_tokens, hedge_candidates[:8])
            emphasis_seeds = {"seriously", "totally", "definitely", "truly", "really", "honestly", "actually", "basically", "pretty"}
            emphasis_candidates = [word for word in self.corpus_profile.common_words if word in emphasis_seeds]
            self.emphasis_tokens = self._merge_unique(self.emphasis_tokens, emphasis_candidates)
            intro_without_commas = [item.rstrip(",").strip().lower() for item in self.corpus_profile.intro_candidates]
            for intro in intro_without_commas:
                if intro and intro not in self.intro_blockers:
                    self.intro_blockers.append(intro)

    def humanize(self, text: str) -> str:
        if not text:
            return text
        normalized = self._normalize(text)
        if not normalized:
            return ""
        paragraphs = self._split_paragraphs(normalized)
        processed: List[str] = []
        for paragraph in paragraphs:
            trimmed = paragraph.strip()
            if not trimmed:
                processed.append("")
                continue
            processed.append(self._humanize_paragraph(trimmed))
        return "\n\n".join(p for p in processed)

    def _normalize(self, text: str) -> str:
        clean = text.replace("\r\n", "\n").replace("\r", "\n")
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r" ?\n ?", "\n", clean)
        return clean.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        return re.split(r"\n\s*\n", text)

    def _humanize_paragraph(self, paragraph: str) -> str:
        sentences = self._split_sentences(paragraph)
        rewritten: List[str] = []
        for index, sentence in enumerate(sentences):
            candidate_list = self._rewrite_sentence(sentence, index, len(sentences))
            for candidate in candidate_list:
                if not candidate:
                    continue
                if (
                    rewritten
                    and len(rewritten[-1]) < 65
                    and len(candidate) < 70
                    and self.rng.random() < self.creativity * 0.45
                    and candidate.split(" ", 1)[0] in self.pair_merge_starters
                ):
                    merged = self._merge_sentences(rewritten[-1], candidate)
                    if merged:
                        rewritten[-1] = merged
                        continue
                rewritten.append(candidate)
        joined = " ".join(rewritten)
        joined = re.sub(r" +", " ", joined)
        joined = re.sub(r" ,", ",", joined)
        joined = re.sub(r" \.", ".", joined)
        joined = re.sub(r" !", "!", joined)
        joined = re.sub(r" \?", "?", joined)
        return joined

    def _split_sentences(self, paragraph: str) -> List[str]:
        pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
        parts = pattern.split(paragraph.strip())
        if len(parts) == 1:
            fallback = re.split(r"(?<=[.!?])\s+", paragraph.strip())
            return [p for p in fallback if p]
        return [p for p in parts if p]

    def _rewrite_sentence(self, sentence: str, index: int, total: int) -> List[str]:
        raw = sentence.strip()
        if not raw:
            return [""]
        original_sentence = raw
        terminal = ""
        if raw[-1] in ".!?":
            terminal = raw[-1]
            raw = raw[:-1]
        core = raw.strip()
        original_core = core
        core = self._apply_phrase_swaps(core)
        core = self._apply_word_swaps(core)
        core = self._structural_variations(core)
        core = self._apply_contractions(core)
        core = self._relax_leading(core)
        core = self._inject_mid_hedges(core)
        core = self._infuse_emphasis(core)
        core = self._personalize_statement(core)
        if self.rng.random() < self.creativity * 0.5:
            core = self._maybe_intro(core, index)
        if terminal not in ("?", "!") and self.rng.random() < self.creativity * 0.35:
            core = self._maybe_outro(core)
        if self.corpus_profile:
            core = self._corpus_guidance(core)
        core = self._tidy_connectors(core)
        core = self._clean_spacing(core)
        sentence_text = self._finalize_sentence(core, terminal or ".")
        if self._equivalent_sentence(sentence_text, original_sentence):
            forced = self._force_variation(original_core, terminal or ".")
            if forced and not self._equivalent_sentence(forced, original_sentence):
                sentence_text = forced
        if self._should_split(sentence_text):
            return self._split_long(sentence_text)
        return [sentence_text]

    def _apply_phrase_swaps(self, text: str) -> str:
        for source, variants in sorted(self._phrase_variants, key=lambda item: len(item[0]), reverse=True):
            text = self._case_variant_replace(text, source, variants)
        return text

    def _apply_word_swaps(self, text: str) -> str:
        for source, variants in sorted(self._word_variants, key=lambda item: len(item[0]), reverse=True):
            text = self._case_variant_replace(text, source, variants)
        return text

    def _apply_contractions(self, text: str) -> str:
        for source, replacement in self._contractions:
            pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
            text = pattern.sub(lambda match: self._match_case(match.group(0), replacement), text)
        return text

    def _relax_leading(self, text: str) -> str:
        leading_match = re.match(r"^([A-Za-z']+)(,\s+)?(.*)$", text)
        if not leading_match:
            return text
        head = leading_match.group(1)
        rest = leading_match.group(3)
        lower_head = head.lower()
        if lower_head in self._formal_leads:
            choice = self._select(self._formal_leads[lower_head])
            return self._match_case(head, choice) + ", " + rest.lstrip()
        return text

    def _inject_mid_hedges(self, text: str) -> str:
        pattern = re.compile(r"\b(I|We|You)\b(\s+)(\b\w+\b)", re.IGNORECASE)
        def replacer(match: re.Match[str]) -> str:
            if self.rng.random() > self.creativity * 0.6:
                return match.group(0)
            pronoun = match.group(1)
            spacer = match.group(2)
            follower = match.group(3)
            insert = self._select(self.hedge_tokens)
            return pronoun + " " + insert + spacer + follower
        return pattern.sub(replacer, text, count=1)

    def _infuse_emphasis(self, text: str) -> str:
        pattern = re.compile(r"\b(really|very|truly|quite|honestly|frankly)\b", re.IGNORECASE)
        if pattern.search(text):
            return text
        if self.rng.random() > self.creativity * 0.5:
            return text
        anchor_pattern = re.compile(r"\b(I|We|It|This|That|They|You)\b", re.IGNORECASE)
        anchor = anchor_pattern.search(text)
        if not anchor:
            return text
        token = self._select(self.emphasis_tokens)
        insertion = anchor.end()
        if insertion < len(text) and text[insertion] == "'":
            while insertion < len(text) and text[insertion] not in {" ", "\t"}:
                insertion += 1
        return text[:insertion] + " " + token + text[insertion:]

    def _maybe_intro(self, text: str, index: int) -> str:
        if self._leading_has_transition(text):
            return text
        stripped = text.lstrip()
        lowered_intro = [lean.lower().rstrip(",") for lean in self.intro_leans]
        for lean in lowered_intro:
            if stripped.lower().startswith(lean):
                return text
        base_chance = min(0.3, 0.15 + self.creativity * 0.2)
        if index == 0 and self.rng.random() < base_chance:
            return self._select(self.intro_leans) + " " + self._lower_after_prefix(text)
        if self.rng.random() < self.creativity * 0.15:
            return self._select(self.intro_leans) + " " + self._lower_after_prefix(text)
        return text

    def _maybe_outro(self, text: str) -> str:
        if len(text) < 25:
            return text
        if text.endswith(("?", "!", "))")):
            return text
        if self.rng.random() > self.creativity * 0.7:
            return text
        tail = self._select(self.outro_tails)
        if text.endswith("."):
            base = text.rstrip(".")
            return base + ", " + tail + "."
        return text + ", " + tail

    def _corpus_guidance(self, text: str) -> str:
        if not self.corpus_profile:
            return text
        adjusted = text
        score = self.corpus_profile.score_text(text)
        if score < 0.18 and self.corpus_mid_phrases and self.rng.random() < 0.7:
            adjusted = self._inject_corpus_phrase(adjusted)
        if score < 0.12 and self.outro_tails and self.rng.random() < self.creativity * 0.4:
            adjusted = self._ensure_tail_phrase(adjusted)
        return adjusted

    def _inject_corpus_phrase(self, text: str) -> str:
        phrase = self._select(self.corpus_mid_phrases)
        if not phrase:
            return text
        lowered_text = text.lower()
        if phrase in lowered_text:
            return text
        if "," in text:
            head, rest = text.split(",", 1)
            remainder = rest.lstrip()
            return f"{head}, {phrase}, {remainder}" if remainder else f"{head}, {phrase}"
        pron_pattern = re.compile(r"\b(I|We|You|They|It|This|That)\b", re.IGNORECASE)
        match = pron_pattern.search(text)
        if match:
            first_word = phrase.split(" ", 1)[0]
            if first_word.lower() == match.group(1).lower():
                remainder = phrase[len(first_word) :].strip()
                if not remainder:
                    return text
                insertion = " " + remainder + ","
                return text[: match.end()] + insertion + text[match.end():]
            pronouns = {"i", "we", "you", "they", "it", "this", "that"}
            if first_word.lower() in pronouns:
                return phrase.capitalize() + ", " + self._lower_after_prefix(text)
            else:
                insertion = " " + phrase + ","
                return text[: match.end()] + insertion + text[match.end():]
        parts = text.split(" ", 1)
        if len(parts) == 2:
            return parts[0] + " " + phrase + ", " + parts[1].lstrip()
        return phrase.capitalize() + ", " + self._lower_after_prefix(text)

    def _ensure_tail_phrase(self, text: str) -> str:
        candidate = self._select(self.outro_tails)
        if not candidate:
            return text
        lowered = text.lower()
        if candidate in lowered:
            return text
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            stripped = stripped.rstrip(".!? ")
        if stripped.endswith((",", ";")):
            return stripped + " " + candidate
        return stripped + ", " + candidate

    def _clean_spacing(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        return text.strip()

    def _equivalent_sentence(self, left: str, right: str) -> bool:
        return self._normalize_sentence(left) == self._normalize_sentence(right)

    def _normalize_sentence(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip().lower()

    def _tidy_connectors(self, text: str) -> str:
        phrases = ["because of that", "as a result", "so that meant", "so that"]
        for phrase in phrases:
            pattern = re.compile(rf"({re.escape(phrase)})(?!,)(?=\s+[A-Za-z])", re.IGNORECASE)
            text = pattern.sub(r"\1,", text)
        return text

    def _force_variation(self, core: str, terminal: str) -> str:
        stripped = core.strip()
        if not stripped:
            return ""
        working = stripped

        link_pattern = re.compile(
            r"\b(?:(?:has|have|had)\s+been|(?:will|would|could|should|might)\s+be|is|are|was|were|be|being|been|seems|feels|looks|remains)\b",
            re.IGNORECASE,
        )

        def add_hedge(match: re.Match[str]) -> str:
            token = self._select(self.hedge_tokens) or "pretty much"
            return match.group(0) + " " + token

        injected, count = link_pattern.subn(add_hedge, working, count=1)
        if count:
            return self._finalize_sentence(self._clean_spacing(injected), terminal)

        pron_pattern = re.compile(r"^(?P<lead>(?:I|We|You|They|It|This|That|These|Those))\b(?P<rest>.*)", re.IGNORECASE)
        pron_match = pron_pattern.match(working)
        if pron_match:
            emphasis = self._select(self.emphasis_tokens) or "honestly"
            rest = pron_match.group("rest").lstrip()
            candidate = pron_match.group("lead") + " " + emphasis
            if rest:
                candidate += " " + rest
            return self._finalize_sentence(self._clean_spacing(candidate), terminal)

        intro = self._select(self.intro_leans) or "Honestly,"
        candidate = intro + " " + self._lower_after_prefix(working)
        return self._finalize_sentence(self._clean_spacing(candidate), terminal)

    def _finalize_sentence(self, text: str, terminal: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if not stripped.endswith(terminal):
            stripped = stripped + terminal
        lead = stripped[0]
        if lead.isalpha():
            stripped = lead.upper() + stripped[1:]
        return stripped

    def _should_split(self, sentence: str) -> bool:
        if len(sentence) < 160:
            return False
        if sentence.count(",") + sentence.count(";") == 0:
            return False
        return self.rng.random() < self.creativity * 0.6

    def _split_long(self, sentence: str) -> List[str]:
        commas = [match.start() for match in re.finditer(r",", sentence)]
        if not commas:
            return [sentence]
        split_point = commas[len(commas) // 2]
        first = sentence[:split_point].rstrip(", ")
        second = sentence[split_point + 1 :].lstrip()
        second = second[0].upper() + second[1:] if second else ""
        if not first.endswith((".", "!", "?")):
            first += "."
        if not second.endswith((".", "!", "?")) and second:
            second += "."
        return [first, second]

    def _merge_sentences(self, first: str, second: str) -> str:
        if not first or not second:
            return ""
        first_core = first.rstrip(".!?")
        second_core = second.strip()
        if not second_core:
            return first
        lower_second = second_core[0].lower() + second_core[1:]
        merged = first_core + ", and " + lower_second
        if not merged.endswith((".", "!", "?")):
            merged += "."
        return merged

    def _personalize_statement(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        removable_tokens = set(self.intro_blockers) | {"frankly", "honestly", "personally", "basically"}
        removed_segments: List[Tuple[str, str]] = []
        core = stripped
        while True:
            comma_index = core.find(",")
            if comma_index == -1:
                break
            candidate_raw = core[:comma_index].strip().lower()
            candidate_clean = re.sub(r"\b(personally|frankly|honestly|genuinely|actually|really)\b", "", candidate_raw)
            candidate_clean = re.sub(r"\s+", " ", candidate_clean).strip()
            if candidate_clean not in removable_tokens:
                break
            next_index = comma_index + 1
            while next_index < len(core) and core[next_index] == " ":
                next_index += 1
            removed_segments.append((core[:next_index], candidate_clean))
            core = core[next_index:]
        core = core.lstrip()
        pattern = re.compile(
            r"^(?P<subject>[A-Za-z0-9 ,'\-]+?)\s+(?P<verb>is|are|was|were)\s+(?P<modifier>(?:pretty|really|quite|fairly|absolutely|basically)\s+)?"
            r"(?P<keyword>essential|necessary|needed|required)\s+(?P<link>for|to)\s+(?P<goal>.+)$",
            re.IGNORECASE,
        )
        match = pattern.match(core)
        link = ""
        if match:
            subject = match.group("subject").strip()
            goal = match.group("goal").strip()
            link = match.group("link").lower()
        else:
            alt_pattern = re.compile(
                r"^(?P<subject>[A-Za-z0-9 ,'\-]+?)\s+has to happen\s+(?P<link>for|to)\s+(?P<goal>.+)$",
                re.IGNORECASE,
            )
            alt_match = alt_pattern.match(core)
            if not alt_match:
                return text
            subject = alt_match.group("subject").strip()
            goal = alt_match.group("goal").strip()
            link = alt_match.group("link").lower()

        subject_phrase = subject
        if subject_phrase and subject_phrase[0].isupper():
            subject_phrase = subject_phrase[0].lower() + subject_phrase[1:]

        goal_phrase = goal
        if goal_phrase and goal_phrase[0].isupper():
            goal_phrase = goal_phrase[0].lower() + goal_phrase[1:]
        goal_phrase = goal_phrase.strip()
        if link == "for":
            lowered_goal = goal_phrase.lower()
            if not lowered_goal.startswith(("the ", "this ", "that ", "our ", "their ", "its ", "any ", "your ")):
                goal_phrase = "the " + goal_phrase

        if link == "to":
            sentence = f"We basically rely on {subject_phrase} to {goal_phrase}"
        else:
            lowered_goal = goal_phrase.lower()
            custom_sentence = ""
            if "compliance" in lowered_goal:
                custom_sentence = f"We basically can't stay compliant without {subject_phrase}"
            elif "success" in lowered_goal:
                custom_sentence = f"We basically can't hit the success we're after without {subject_phrase}"
            elif "performance" in lowered_goal:
                custom_sentence = f"We basically can't hit the performance we're after without {subject_phrase}"
            elif "trust" in lowered_goal:
                custom_sentence = f"We basically can't keep folks trusting us without {subject_phrase}"
            if custom_sentence:
                sentence = custom_sentence
            else:
                qualifier = ""
                if self.rng.random() < self.creativity * 0.4:
                    qualifier = " we're after"
                sentence = f"We basically can't reach {goal_phrase}{qualifier} without {subject_phrase}"

        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        prefix_text = ""
        if removed_segments:
            prefix_token = removed_segments[0][1]
            prefix_map = {
                "still": "Still, ",
                "even so": "Even so, ",
                "even then": "Even then, ",
                "that said": "That said, ",
                "so": "So, ",
                "because of that": "Because of that, ",
                "plus": "Plus, ",
                "what's more": "Plus, ",
            }
            prefix_text = prefix_map.get(prefix_token, "")
        if prefix_text:
            sentence = prefix_text + sentence[0].lower() + sentence[1:]
        return sentence

    def _structural_variations(self, text: str) -> str:
        updated = text
        updated = self._rewrite_it_is_to(updated)
        updated = self._rewrite_passive_need(updated)
        updated = self._rewrite_allows_construct(updated)
        updated = self._rewrite_there_construct(updated)
        return updated

    def _rewrite_it_is_to(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        importance = {
            "important",
            "essential",
            "critical",
            "necessary",
            "vital",
            "key",
            "helpful",
            "useful",
            "smart",
            "valuable",
            "worthwhile",
            "needed",
        }
        pattern_to = re.compile(
            r"^(?P<subject>It|This|That)\s+is\s+(?P<modifier>(?:really|very|quite|pretty|fairly|absolutely|extremely|super|highly|particularly|critically)\s+)?(?P<adjective>[A-Za-z\-']+)\s+to\s+(?P<action>.+)$",
            re.IGNORECASE,
        )
        match_to = pattern_to.match(stripped)
        if match_to:
            adjective = match_to.group("adjective").lower()
            if adjective not in importance:
                return text
            action = match_to.group("action").strip()
            if not action:
                return text
            lowered_action = action.lower()
            pronoun = "We"
            if lowered_action.startswith("your ") or " your " in lowered_action:
                pronoun = "You"
            elif lowered_action.startswith("my ") or " my " in lowered_action:
                pronoun = "I"
            rebuilt = self._assemble_need_statement(pronoun, action)
            if not rebuilt:
                return text
            return self._restore_spacing(text, rebuilt)
        pattern_that = re.compile(
            r"^(?P<subject>It|This|That)\s+is\s+(?P<modifier>(?:really|very|quite|pretty|fairly|absolutely|extremely|super|highly|particularly|critically)\s+)?(?P<adjective>[A-Za-z\-']+)\s+that\s+(?P<clause>.+)$",
            re.IGNORECASE,
        )
        match_that = pattern_that.match(stripped)
        if not match_that:
            return text
        adjective = match_that.group("adjective").lower()
        if adjective not in importance:
            return text
        clause = match_that.group("clause").strip()
        if not clause:
            return text
        lowered_clause = clause.lower()
        pronoun = "We"
        if lowered_clause.startswith("we "):
            clause = clause[3:].lstrip()
            pronoun = "We"
        elif lowered_clause.startswith("you "):
            clause = clause[4:].lstrip()
            pronoun = "You"
        elif lowered_clause.startswith("i "):
            clause = clause[2:].lstrip()
            pronoun = "I"
        else:
            return text
        rebuilt = self._assemble_need_statement(pronoun, clause)
        if not rebuilt:
            return text
        return self._restore_spacing(text, rebuilt)

    def _assemble_need_statement(self, pronoun: str, action: str) -> str:
        cleaned = action.strip()
        if not cleaned:
            return ""
        tokens = cleaned.split()
        if not tokens:
            return ""
        determiner_tokens = {"the", "a", "an", "our", "their", "your", "this", "that", "these", "those", "any", "each", "some"}
        base_pool = self.casual_need_plain if tokens[0].lower() in determiner_tokens else self.casual_need_to
        chunk = self._select(base_pool)
        lowered = cleaned.lower()
        if pronoun == "We" and lowered.startswith("we "):
            cleaned = cleaned[len(tokens[0]) + 1 :].lstrip()
        elif pronoun == "You" and lowered.startswith("you "):
            cleaned = cleaned[len(tokens[0]) + 1 :].lstrip()
        elif pronoun == "I" and lowered.startswith("i "):
            cleaned = cleaned[len(tokens[0]) + 1 :].lstrip()
        if not cleaned:
            return ""
        body = self._lower_after_prefix(cleaned) if base_pool is self.casual_need_to else cleaned
        rebuilt = f"{pronoun} {chunk} {body}".strip()
        return self._clean_spacing(rebuilt)

    def _rewrite_there_construct(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        pattern = re.compile(r"^there\s+(?P<verb>is|are)\s+(?P<body>.+)$", re.IGNORECASE)
        match = pattern.match(stripped)
        if not match:
            return text
        body = match.group("body").strip()
        if not body:
            return text
        verb = match.group("verb").lower()
        lowered_body = body.lower()

        def adjust(fragment: str) -> str:
            if len(fragment) > 1 and fragment[0].isupper() and fragment[1].isupper():
                return fragment
            return self._lower_after_prefix(fragment)

        if verb == "is":
            if lowered_body.startswith("no "):
                remainder = body[3:].lstrip()
                if not remainder:
                    return text
                prefix = self._select(["There's really no", "There's basically no", "There's hardly any"])
                rebuilt = f"{prefix} {remainder}"
            else:
                prefix = self._select(["There's", "There's kind of"])
                rebuilt = f"{prefix} {adjust(body)}"
        else:
            tokens = body.split()
            if tokens and tokens[0].lower() == "still":
                remainder = " ".join(tokens[1:]).strip()
                if not remainder:
                    return text
                choice = self._select(self.casual_still_have_variants)
                rebuilt = f"{choice} {adjust(remainder)}"
            else:
                choice = self._select(self.casual_have_variants)
                rebuilt = f"{choice} {adjust(body)}"
        rebuilt = self._clean_spacing(rebuilt)
        return self._restore_spacing(text, rebuilt)

    def _rewrite_passive_need(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        pattern = re.compile(r"^(?P<object>.+?)\s+needs\s+to\s+be\s+(?P<verb>[A-Za-z]+)(?P<tail>.*)$", re.IGNORECASE)
        match = pattern.match(stripped)
        if not match:
            return text
        verb = match.group("verb")
        lowered = verb.lower()
        if not lowered.endswith(("ed", "en", "wn", "ized", "ised")):
            return text
        obj = match.group("object").strip()
        if not obj:
            return text
        lowered_object = obj.lower()
        if lowered_object.startswith("we ") or lowered_object.startswith("i ") or lowered_object.startswith("you "):
            return text
        tail = match.group("tail") or ""
        helper = self._select(["We need to get", "We should get", "We have to get"])
        obj_body = obj
        if len(obj_body) > 1 and obj_body[0].isupper() and obj_body[1].islower():
            obj_body = obj_body[0].lower() + obj_body[1:]
        rebuilt = f"{helper} {obj_body} {verb}{tail}".strip()
        rebuilt = self._clean_spacing(rebuilt)
        return self._restore_spacing(text, rebuilt)

    def _rewrite_allows_construct(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        pattern = re.compile(
            r"^(?P<lead>This|That|It)\s+(?P<verb>allows|allow|enabled|enables|lets|let|helps)\s+(?P<object>us|me|our team|the team|folks|people|users|customers|everyone|clients)\s+to\s+(?P<action>.+)$",
            re.IGNORECASE,
        )
        match = pattern.match(stripped)
        if not match:
            return text
        action = match.group("action").strip()
        if not action:
            return text
        group_original = match.group("object")
        lowered_group = group_original.lower()
        if lowered_group in {"us", "our team", "the team"}:
            opener = self._select(self.allows_we_variants)
            rebuilt = f"{opener} {self._lower_after_prefix(action)}"
        elif lowered_group == "me":
            opener = self._select(self.allows_me_variants)
            rebuilt = f"{opener} {self._lower_after_prefix(action)}"
        else:
            template = self._select(self.allows_generic_variants)
            opener = template.replace("{group}", group_original)
            rebuilt = f"{opener} {self._lower_after_prefix(action)}"
        rebuilt = self._clean_spacing(rebuilt)
        return self._restore_spacing(text, rebuilt)

    def _leading_has_transition(self, text: str) -> bool:
        stripped = text.strip().lower()
        if not stripped:
            return False
        for blocker in self.intro_blockers:
            if stripped.startswith(blocker):
                return True
            if stripped.startswith(blocker + ","):
                return True
        return False

    def _case_variant_replace(self, text: str, source: str, variants: List[str]) -> str:
        pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
        def repl(match: re.Match[str]) -> str:
            choice = self._select(variants)
            return self._match_case(match.group(0), choice)
        return pattern.sub(repl, text)

    def _match_case(self, original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    def _select(self, options: List[str]) -> str:
        if not options:
            return ""
        if len(options) == 1 or self.creativity < 0.2:
            return options[0]
        index = int(self.rng.random() ** (1.0 - self.creativity) * len(options))
        return options[min(index, len(options) - 1)]

    def _merge_unique(self, base: List[str], extra: Iterable[str]) -> List[str]:
        seen = {item.strip().lower(): True for item in base if item.strip()}
        for item in extra:
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                base.append(cleaned)
                seen[key] = True
        return base

    def _lower_after_prefix(self, text: str) -> str:
        if not text:
            return text
        if text[0].isupper() and len(text) > 1:
            lower_head = text[0].lower() + text[1:]
            return lower_head
        return text

    def _restore_spacing(self, original: str, replacement: str) -> str:
        if not original:
            return replacement
        prefix_len = len(original) - len(original.lstrip())
        suffix_start = len(original.rstrip())
        return original[:prefix_len] + replacement + original[suffix_start:]
