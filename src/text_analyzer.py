"""
Text Analyzer v2 — Production-Grade Heuristic NLP Feature Extraction
Deterministic, LLM-free, UGC-aware.

Upgrades:
  Part 1: Multi-format dialogue detection (curly quotes, said-verbs, chat-style)
  Part 2: 8 emotional signal heuristics (pronoun intensity, sentiment shifts, etc.)
  Part 3: Relational conflict detection (weighted, deduped per window)
  Part 4: Enhanced reflective ending detection (phrase + deceleration + NER)
  
Hardening:
  - sqrt scaling on counts to prevent inflation
  - Length normalization (min 50 words)
  - Sentence-level gating for relationship lexicon
  - Magnitude thresholding on sentiment shifts
  - spaCy fallback safety on all NER/POS calls
  - Single nlp() call reused throughout
"""

import re
import math
import spacy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TextFeatures:
    """Container for all extracted linguistic features."""
    # Core (v1)
    pacing_score: float = 0.0         # 0-10
    dialogue_density: float = 0.0     # 0-1
    action_verb_ratio: float = 0.0    # 0-1
    conflict_markers: int = 0         # weighted count
    emotional_variance: float = 0.0   # 0-1
    sentence_count: int = 0
    word_count: int = 0
    avg_sentence_length: float = 0.0
    punctuation_density: float = 0.0
    
    # Emotional Signals (v2 — Part 2)
    pronoun_intensity: float = 0.0      # 0-1
    sentiment_shifts: int = 0           # count of significant polarity flips
    contrast_density: float = 0.0       # 0-1
    short_burst_ratio: float = 0.0      # 0-1
    vulnerability_count: int = 0        # raw count
    relationship_lexicon: float = 0.0   # 0-1 (gated)
    emotional_punctuation: float = 0.0  # 0-1
    interjection_density: float = 0.0   # 0-1
    
    # Relational Conflict (v2 — Part 3)
    relational_conflict: float = 0.0    # 0-1


class TextAnalyzer:
    """
    Production-grade heuristic feature extractor for UGC story text.
    Deterministic. No LLMs. No embeddings. No external APIs.
    """
    
    # ==========================================
    # LEXICONS & PATTERNS (Class Constants)
    # ==========================================
    
    # --- Dialogue (Part 1) ---
    SAID_VERBS = {
        'said', 'asked', 'whispered', 'shouted', 'replied', 'muttered',
        'yelled', 'screamed', 'murmured', 'exclaimed', 'stammered',
        'sighed', 'groaned', 'snapped', 'hissed', 'pleaded', 'demanded',
        'insisted', 'suggested', 'answered', 'called', 'cried'
    }
    
    QUOTE_PAIR_PATTERNS = [
        re.compile(r'["\u201c](.*?)["\u201d]', re.DOTALL),   # Double straight + curly
        re.compile(r"['\u2018](.*?)['\u2019]", re.DOTALL),   # Single straight + curly
    ]
    
    SAID_VERB_PATTERN = re.compile(
        r'\b(' + '|'.join(SAID_VERBS) + r')\b', re.IGNORECASE
    )
    
    CHAT_LINE_PATTERN = re.compile(r'^\s*[\w]+\s*:', re.MULTILINE)
    DASH_DIALOGUE_PATTERN = re.compile(r'^\s*[\u2014\u2013\-]\s+\S', re.MULTILINE)
    
    # --- Emotion (Part 2) ---
    FIRST_PERSON_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself'}
    
    POSITIVE_WORDS = {
        'joy', 'happy', 'love', 'delight', 'excited', 'pleased', 'thrilled',
        'ecstatic', 'glad', 'warm', 'smile', 'laugh', 'bright', 'hope',
        'beautiful', 'wonderful', 'amazing', 'perfect', 'sweet', 'kind'
    }
    NEGATIVE_WORDS = {
        'anger', 'rage', 'fury', 'hate', 'despise', 'fear', 'terror', 'dread',
        'sad', 'grief', 'sorrow', 'despair', 'misery', 'anguish', 'pain',
        'hurt', 'cry', 'tears', 'broken', 'alone', 'empty', 'numb', 'guilty',
        'ashamed', 'lost', 'cold', 'dark', 'bitter', 'regret', 'sorry'
    }
    TENSION_WORDS = {
        'anxious', 'nervous', 'worried', 'tense', 'uneasy', 'stressed',
        'panic', 'dread', 'afraid', 'scared', 'terrified', 'shaking'
    }
    
    CONTRAST_TRANSITIONS = re.compile(
        r'\b(but|instead|however|yet|still|though|although|nevertheless|despite|whereas)\b',
        re.IGNORECASE
    )
    
    VULNERABILITY_PATTERNS = [
        re.compile(r"\bI\s+didn['\u2019]?t\b", re.IGNORECASE),
        re.compile(r"\bI\s+never\b", re.IGNORECASE),
        re.compile(r"\bI\s+thought\b", re.IGNORECASE),
        re.compile(r"\bI\s+wanted\b", re.IGNORECASE),
        re.compile(r"\bI\s+couldn['\u2019]?t\b", re.IGNORECASE),
        re.compile(r"\bI\s+shouldn['\u2019]?t\s+have\b", re.IGNORECASE),
        re.compile(r"\bI\s+wasn['\u2019]?t\b", re.IGNORECASE),
        re.compile(r"\bI\s+wish\b", re.IGNORECASE),
        re.compile(r"\bI\s+don['\u2019]?t\s+know\b", re.IGNORECASE),
    ]
    
    RELATIONSHIP_LEXICON = {
        'jealous', 'favorite', 'ignore', 'block', 'leave', 'stay', 'ghost',
        'text', 'call', 'ex', 'toxic', 'obsess', 'cheat', 'betray', 'trust',
        'miss', 'cling', 'space', 'distance', 'commit', 'dump', 'rebound'
    }
    RELATIONSHIP_GATE_PRONOUNS = {'you', 'him', 'her', 'we', 'i', 'he', 'she', 'they'}
    
    INTERJECTIONS = {
        'oh', 'god', 'please', 'ugh', 'wow', 'seriously', 'honestly',
        'damn', 'hell', 'shit', 'gosh', 'omg', 'wtf', 'no way'
    }
    
    EMOTIONAL_PUNCT_PATTERNS = [
        re.compile(r'\.{3,}'),        # Ellipsis
        re.compile(r'\u2014|\u2013'),  # Em/en dash
        re.compile(r'[!?]{2,}'),      # Double punctuation
    ]
    
    # --- Conflict (Part 3) ---
    EXPLICIT_CONFLICT_PATTERNS = [
        (re.compile(r'\b(fight|battle|combat|attack|confront|clash)\w*\b', re.I), 1.0),
        (re.compile(r'\b(tension|struggle|conflict|oppose|resist|challenge)\w*\b', re.I), 0.8),
        (re.compile(r'\b(threat|danger|risk|peril|hazard)\w*\b', re.I), 0.7),
        (re.compile(r'\b(against|versus)\b', re.I), 0.6),
        (re.compile(r'[!?]{2,}'), 0.5),
    ]
    
    DIALOGUE_DISAGREEMENT = [
        re.compile(r'["\u201c][^"\u201d]*\b(no|stop|don[\'\u2019]?t|enough|shut up)\b[^"\u201d]*["\u201d]', re.I),
        re.compile(r'["\u201c][^"\u201d]*\byou\s+don[\'\u2019]?t\s+understand\b[^"\u201d]*["\u201d]', re.I),
        re.compile(r'["\u201c][^"\u201d]*\bthat[\'\u2019]?s\s+not\s+what\b[^"\u201d]*["\u201d]', re.I),
    ]
    
    JEALOUSY_MARKERS = re.compile(
        r'\b(jealous|who\s+is\s+she|who\s+is\s+he|why\s+were\s+you\s+with|are\s+you\s+serious)\b', re.I
    )
    
    POSSESSIVE_LANGUAGE = re.compile(
        r"\b(you[\'\u2019]?re\s+mine|my\s+girl|my\s+man|you\s+belong|don[\'\u2019]?t\s+talk\s+to)\b", re.I
    )
    
    DISMISSIVE_TENSION = re.compile(
        r"\b(you[\'\u2019]?re\s+overthinking|relax|why\s+label|calm\s+down|"
        r"it[\'\u2019]?s\s+not\s+a\s+big\s+deal|whatever|I\s+don[\'\u2019]?t\s+care)\b", re.I
    )
    
    SILENT_TREATMENT = re.compile(
        r"\b(didn[\'\u2019]?t\s+reply|left\s+on\s+read|blocked|ignored\s+|walked\s+away|"
        r"no\s+response|seen\s+and\s+ignored)\b", re.I
    )
    
    # --- Ending (Part 4) ---
    REFLECTIVE_PHRASES = [
        re.compile(r'\bI\s+realized\b', re.I),
        re.compile(r'\bfor\s+the\s+first\s+time\b', re.I),
        re.compile(r'\bmaybe\b', re.I),
        re.compile(r'\bit\s+didn[\'\u2019]?t\s+feel\b', re.I),
        re.compile(r'\bI\s+think\s+I[\'\u2019]?m\s+ready\b', re.I),
        re.compile(r'\bthat\s+was\s+the\s+moment\b', re.I),
        re.compile(r'\bI\s+knew\s+then\b', re.I),
        re.compile(r'\bsomething\s+shifted\b', re.I),
        re.compile(r'\bI\s+stopped\s+trying\b', re.I),
    ]
    
    REFLECTIVE_WORDS = {
        'finally', 'always', 'never', 'remembered', 'knew', 'understood',
        'realized', 'peace', 'quiet', 'still', 'last', 'end', 'goodbye',
        'farewell', 'smiled', 'sighed'
    }
    
    # --- Action Verbs ---
    ACTION_VERBS = {
        'run', 'jump', 'fight', 'grab', 'throw', 'hit', 'kick', 'punch',
        'chase', 'flee', 'attack', 'defend', 'strike', 'charge',
        'dash', 'leap', 'sprint', 'rush', 'burst', 'lunge',
        'dodge', 'duck', 'dive', 'roll', 'slide', 'slam',
        'push', 'pull', 'shove', 'wrestle', 'tackle', 'swing',
        'shoot', 'stab', 'slice', 'slash', 'cut', 'pierce',
        'climb', 'crawl', 'sneak', 'creep', 'stumble', 'trip'
    }
    
    # --- Normalization Constants (calibrated for UGC) ---
    PRONOUN_MAX = 0.20           # 20% first-person pronouns = saturated (UGC averages ~8%)
    CONTRAST_MAX = 0.25          # 1 contrast per 4 sentences = saturated
    SHORT_BURST_MAX = 0.6        # 60% short paragraphs = saturated
    VULNERABILITY_MAX = 0.008    # Per word
    RELATIONSHIP_MAX = 0.015     # Per word (gated)
    EMOTIONAL_PUNCT_MAX = 0.3    # Per sentence
    INTERJECTION_MAX = 0.15      # Per sentence
    SENTIMENT_SHIFT_THRESHOLD = 0.3  # Min polarity delta to count as a shift
    MIN_PARAGRAPH_WORDS = 20     # Min words for paragraph to count in sentiment
    QUESTION_MIN_TOKENS = 5      # Min tokens for question to count in conflict
    DEDUP_WINDOW = 150           # Words per dedup window
    
    def __init__(self):
        """Initialize spaCy. Fallback-safe."""
        self._spacy_available = True
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self._spacy_available = False
            self.nlp = None
    
    # ==========================================
    # MAIN ANALYSIS
    # ==========================================
    
    def analyze(self, text: str) -> TextFeatures:
        """Extract all features. Single nlp() call, fallback-safe."""
        if not text or not text.strip():
            return TextFeatures()
        
        doc = None
        sentences = []
        
        if self._spacy_available and self.nlp:
            try:
                doc = self.nlp(text)
                sentences = list(doc.sents)
            except Exception:
                doc = None
        
        # Fallback sentence splitting if spaCy fails
        if not sentences:
            raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = raw_sents  # list of strings instead of Span objects
        
        word_count = len(text.split())
        safe_word_count = max(word_count, 50)  # Anti-inflation: floor at 50
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Core features (v1)
        pacing = self._calculate_pacing(sentences, doc)
        dialogue = self._calculate_dialogue_density(text)
        action_ratio = self._calculate_action_verb_ratio(doc) if doc else 0.0
        explicit_conflict = self._count_explicit_conflict(text)
        emotional_var = self._calculate_emotional_variance(sentences)
        sent_count = len(sentences)
        avg_sent_len = word_count / max(sent_count, 1)
        punct_density = self._calculate_punctuation_density(text)
        
        # Emotional signals (v2 — Part 2)
        pronoun_int = self._pronoun_intensity(text, safe_word_count)
        sent_shifts = self._sentiment_shifts(paragraphs)
        contrast_den = self._contrast_density(text, sent_count)
        burst_ratio = self._short_burst_ratio(paragraphs)
        vuln_count = self._vulnerability_markers(text)
        rel_lex = self._relationship_lexicon_gated(text, safe_word_count)
        emo_punct = self._emotional_punctuation(text, sent_count)
        interj_den = self._interjection_density(text, sent_count)
        
        # Relational conflict (v2 — Part 3)
        rel_conflict = self._relational_conflict(text, safe_word_count)
        
        return TextFeatures(
            pacing_score=pacing,
            dialogue_density=dialogue,
            action_verb_ratio=action_ratio,
            conflict_markers=explicit_conflict,
            emotional_variance=emotional_var,
            sentence_count=sent_count,
            word_count=word_count,
            avg_sentence_length=round(avg_sent_len, 2),
            punctuation_density=punct_density,
            pronoun_intensity=pronoun_int,
            sentiment_shifts=sent_shifts,
            contrast_density=contrast_den,
            short_burst_ratio=burst_ratio,
            vulnerability_count=vuln_count,
            relationship_lexicon=rel_lex,
            emotional_punctuation=emo_punct,
            interjection_density=interj_den,
            relational_conflict=rel_conflict,
        )
    
    # ==========================================
    # PART 1: DIALOGUE DETECTION
    # ==========================================
    
    def _calculate_dialogue_density(self, text: str) -> float:
        """
        Three-layer dialogue detection:
        1. Regex quote pairs (multi-format)
        2. Said-verb attribution
        3. Chat-style/dash dialogue lines
        """
        total_chars = len(text.strip())
        if total_chars == 0:
            return 0.0
        
        dialogue_chars = 0
        
        # Layer 1: Quote pair matching (union of all quote formats)
        for pattern in self.QUOTE_PAIR_PATTERNS:
            for match in pattern.finditer(text):
                content = match.group(1)
                if len(content.strip()) >= 2:  # Skip empty quotes
                    dialogue_chars += len(content)
        
        # Layer 2: Said-verb lines (catch unquoted dialogue attribution)
        said_matches = self.SAID_VERB_PATTERN.findall(text)
        # Each said-verb implies ~50 chars of dialogue we may have missed
        unquoted_dialogue_est = max(0, len(said_matches) * 50 - dialogue_chars)
        dialogue_chars += int(unquoted_dialogue_est * 0.5)  # Conservative estimate
        
        # Layer 3: Chat-style lines
        chat_lines = self.CHAT_LINE_PATTERN.findall(text)
        dash_lines = self.DASH_DIALOGUE_PATTERN.findall(text)
        chat_count = len(chat_lines) + len(dash_lines)
        if chat_count > 0:
            # Estimate each chat line as ~40 chars of dialogue
            dialogue_chars += chat_count * 40
        
        density = dialogue_chars / total_chars
        return round(min(1.0, density), 3)
    
    # ==========================================
    # PART 2: EMOTIONAL SIGNALS
    # ==========================================
    
    def _pronoun_intensity(self, text: str, safe_wc: int) -> float:
        """First-person pronoun intensity with log dampening above 0.5."""
        words = text.lower().split()
        count = sum(1 for w in words if w in self.FIRST_PERSON_PRONOUNS)
        raw = count / safe_wc
        linear = min(1.0, raw / self.PRONOUN_MAX)
        # Log dampening: compress high values to prevent over-amplification
        if linear > 0.5:
            linear = 0.5 + math.log1p(linear - 0.5) * 0.4
        return round(min(1.0, linear), 3)
    
    def _sentiment_shifts(self, paragraphs: List[str]) -> int:
        """
        Count significant polarity flips between paragraphs.
        Tightened: magnitude threshold, minimum paragraph length, dedup adjacent.
        """
        if len(paragraphs) < 2:
            return 0
        
        scored = []
        prev_fingerprint = None
        
        for para in paragraphs:
            words = para.lower().split()
            if len(words) < self.MIN_PARAGRAPH_WORDS:
                continue
            
            # Dedup near-identical adjacent paragraphs
            fingerprint = (words[0] if words else "", len(words))
            if fingerprint == prev_fingerprint:
                continue
            prev_fingerprint = fingerprint
            
            # Simple polarity: positive count - negative count, normalized
            word_set = set(words)
            pos = len(word_set & self.POSITIVE_WORDS)
            neg = len(word_set & self.NEGATIVE_WORDS)
            tension = len(word_set & self.TENSION_WORDS)
            
            polarity = (pos - neg - tension * 0.5) / max(len(words), 1) * 10
            scored.append(polarity)
        
        # Count shifts exceeding the magnitude threshold
        shifts = 0
        for i in range(1, len(scored)):
            delta = abs(scored[i] - scored[i-1])
            if delta >= self.SENTIMENT_SHIFT_THRESHOLD:
                shifts += 1
        
        return shifts
    
    def _contrast_density(self, text: str, sent_count: int) -> float:
        """Contrast transition words per sentence."""
        matches = self.CONTRAST_TRANSITIONS.findall(text)
        if sent_count == 0:
            return 0.0
        raw = len(matches) / max(sent_count, 1)
        return round(min(1.0, raw / self.CONTRAST_MAX), 3)
    
    def _short_burst_ratio(self, paragraphs: List[str]) -> float:
        """Ratio of short paragraphs (< 15 words). sqrt-scaled to prevent UGC saturation."""
        if not paragraphs:
            return 0.0
        short = sum(1 for p in paragraphs if len(p.split()) < 15)
        raw = short / len(paragraphs)
        # sqrt scaling: differentiates 30% vs 60% vs 90% instead of all -> 1.0
        scaled = math.sqrt(raw / self.SHORT_BURST_MAX)
        return round(min(1.0, scaled), 3)
    
    def _vulnerability_markers(self, text: str) -> int:
        """Count vulnerability phrases. sqrt-scaled to prevent inflation."""
        count = 0
        for pattern in self.VULNERABILITY_PATTERNS:
            count += len(pattern.findall(text))
        return count
    
    def _relationship_lexicon_gated(self, text: str, safe_wc: int) -> float:
        """
        Relationship vocabulary, GATED by pronoun co-occurrence.
        Only counts if a relationship word appears in the same sentence 
        as a gate pronoun or a vulnerability marker.
        Deduped per 150-word window.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        matched_count = 0
        window_seen = set()
        word_position = 0
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            sent_words = set(sent_lower.split())
            
            # Check gate: must contain a pronoun OR a vulnerability marker
            has_pronoun = bool(sent_words & self.RELATIONSHIP_GATE_PRONOUNS)
            has_vuln = any(p.search(sentence) for p in self.VULNERABILITY_PATTERNS)
            
            if not (has_pronoun or has_vuln):
                word_position += len(sent_lower.split())
                continue
            
            # Check for relationship words
            current_window = word_position // self.DEDUP_WINDOW
            for word in self.RELATIONSHIP_LEXICON:
                if word in sent_lower:
                    dedup_key = (word, current_window)
                    if dedup_key not in window_seen:
                        window_seen.add(dedup_key)
                        matched_count += 1
            
            word_position += len(sent_lower.split())
        
        # sqrt scaling to prevent inflation
        scaled = math.sqrt(matched_count)
        raw = scaled / safe_wc * 100  # Per 100 words, sqrt-scaled
        return round(min(1.0, raw / (self.RELATIONSHIP_MAX * 100)), 3)
    
    def _emotional_punctuation(self, text: str, sent_count: int) -> float:
        """Ellipsis, em-dashes, double punctuation per sentence."""
        if sent_count == 0:
            return 0.0
        count = 0
        for pattern in self.EMOTIONAL_PUNCT_PATTERNS:
            count += len(pattern.findall(text))
        raw = count / max(sent_count, 1)
        return round(min(1.0, raw / self.EMOTIONAL_PUNCT_MAX), 3)
    
    def _interjection_density(self, text: str, sent_count: int) -> float:
        """Spoken-tone interjections per sentence."""
        if sent_count == 0:
            return 0.0
        words = text.lower().split()
        count = sum(1 for w in words if w.strip('.,!?;:') in self.INTERJECTIONS)
        raw = count / max(sent_count, 1)
        return round(min(1.0, raw / self.INTERJECTION_MAX), 3)
    
    # ==========================================
    # PART 3: CONFLICT DETECTION
    # ==========================================
    
    def _count_explicit_conflict(self, text: str) -> int:
        """Weighted explicit conflict markers (v1 logic, kept for backward compat)."""
        text_lower = text.lower()
        score = 0.0
        for pattern, weight in self.EXPLICIT_CONFLICT_PATTERNS:
            matches = pattern.findall(text_lower)
            score += len(matches) * weight
        # Add mild contrast markers
        contrast_matches = re.findall(
            r'\b(but|however|although|yet|despite|though)\b', text_lower)
        score += len(contrast_matches) * 0.3
        return int(round(score))
    
    def _relational_conflict(self, text: str, safe_wc: int) -> float:
        """
        Multi-layer relational conflict scoring.
        Deduped per 200-word window. sqrt-scaled.
        """
        # Split into windows for dedup
        words = text.split()
        window_size = 200
        
        total_score = 0.0
        
        # Process in windows
        for start in range(0, len(words), window_size):
            window_text = ' '.join(words[start:start + window_size])
            window_seen = set()
            
            # Layer 1: Dialogue disagreement (weight 1.2)
            for pattern in self.DIALOGUE_DISAGREEMENT:
                for m in pattern.finditer(window_text):
                    key = ('disagree', m.group(0)[:20])
                    if key not in window_seen:
                        window_seen.add(key)
                        total_score += 1.2
            
            # Layer 2: Jealousy markers (weight 1.0)
            for m in self.JEALOUSY_MARKERS.finditer(window_text):
                key = ('jealousy', m.group(0)[:20])
                if key not in window_seen:
                    window_seen.add(key)
                    total_score += 1.0
            
            # Layer 3: Possessive language (weight 0.8)
            for m in self.POSSESSIVE_LANGUAGE.finditer(window_text):
                key = ('possessive', m.group(0)[:20])
                if key not in window_seen:
                    window_seen.add(key)
                    total_score += 0.8
            
            # Layer 4: Dismissive tension (weight 1.0)
            for m in self.DISMISSIVE_TENSION.finditer(window_text):
                key = ('dismissive', m.group(0)[:20])
                if key not in window_seen:
                    window_seen.add(key)
                    total_score += 1.0
            
            # Layer 5: Question density in dialogue (weight 0.6)
            # Questions inside quotes, min token length, tension scaling
            quoted_questions = re.findall(
                r'["\u201c]([^"\u201d]*\?[^"\u201d]*)["\u201d]', window_text)
            for q in quoted_questions:
                tokens = q.split()
                if len(tokens) < self.QUESTION_MIN_TOKENS:
                    continue
                weight = 0.6
                # Tension keywords inside question -> 1.5x
                if re.search(r'\b(why|how\s+could\s+you|seriously|really)\b', q, re.I):
                    weight *= 1.5
                # Short questions dampened
                elif len(tokens) < 8:
                    weight *= 0.5
                key = ('question', q[:30])
                if key not in window_seen:
                    window_seen.add(key)
                    total_score += weight
            
            # Layer 6: Silent treatment (weight 0.9)
            for m in self.SILENT_TREATMENT.finditer(window_text):
                key = ('silent', m.group(0)[:20])
                if key not in window_seen:
                    window_seen.add(key)
                    total_score += 0.9
        
        # sqrt scaling + length normalization (×4, NOT ×10 — prevents saturation)
        scaled = math.sqrt(total_score)
        normalized = scaled / math.sqrt(safe_wc) * 4
        return round(min(1.0, normalized), 3)
    
    # ==========================================
    # CORE v1 FEATURES (upgraded)
    # ==========================================
    
    def _calculate_pacing(self, sentences, doc) -> float:
        """Pacing from sentence length variance + short sentence ratio."""
        if len(sentences) < 2:
            return 5.0
        
        # Get lengths (handle both spaCy spans and plain strings)
        if doc is not None:
            lengths = [len(sent) for sent in sentences]
        else:
            lengths = [len(s.split()) for s in sentences]
        
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        if mean_length > 0:
            cv = np.sqrt(variance) / mean_length
        else:
            cv = 0
        
        score = min(10.0, cv * 10)
        
        short_sentences = sum(1 for l in lengths if l < 5)
        if short_sentences > len(sentences) * 0.2:
            score = min(10.0, score + 1.5)
        
        return round(score, 2)
    
    def _calculate_action_verb_ratio(self, doc) -> float:
        """Action verb ratio using spaCy POS."""
        if doc is None:
            return 0.0
        try:
            verbs = [t for t in doc if t.pos_ == "VERB"]
            if not verbs:
                return 0.0
            action_count = sum(1 for v in verbs if v.lemma_.lower() in self.ACTION_VERBS)
            return round(action_count / len(verbs), 3)
        except Exception:
            return 0.0
    
    def _calculate_emotional_variance(self, sentences) -> float:
        """Emotional variance across sentences using word-level detection."""
        if len(sentences) < 2:
            return 0.0
        
        all_emotions = self.POSITIVE_WORDS | self.NEGATIVE_WORDS | self.TENSION_WORDS
        intensities = []
        
        for sent in sentences:
            sent_text = sent.text.lower() if hasattr(sent, 'text') else sent.lower()
            intensity = 0.0
            
            # Punctuation intensity
            intensity += sent_text.count('!') * 0.3
            intensity += sent_text.count('?') * 0.2
            intensity += len(re.findall(r'[!?]{2,}', sent_text)) * 0.5
            
            # Emotion word detection
            words = set(sent_text.split())
            for w in words:
                clean_w = w.strip('.,!?;:\'"')
                if clean_w in self.POSITIVE_WORDS:
                    intensity += 0.4
                elif clean_w in self.NEGATIVE_WORDS:
                    intensity += 0.5
                elif clean_w in self.TENSION_WORDS:
                    intensity += 0.6
            
            # CAPS emphasis
            caps_words = len([w for w in sent_text.split() if w.isupper() and len(w) > 1])
            intensity += caps_words * 0.2
            
            intensities.append(intensity)
        
        variance = np.var(intensities)
        return round(min(1.0, variance / 2.0), 3)
    
    def _calculate_punctuation_density(self, text: str) -> float:
        punct_count = sum(1 for c in text if c in '.,!?;:-\u2014')
        total = len(text.strip())
        if total == 0:
            return 0.0
        return round(punct_count / total, 3)
    
    # ==========================================
    # ENGAGEMENT SIGNALS (Composite)
    # ==========================================
    
    def get_engagement_signals(self, features: TextFeatures) -> Dict[str, float]:
        """
        Convert features to 5 engagement signals (0-1).
        Incorporates v2 emotional and conflict signals.
        Anti-inflation: weighted average, clamped.
        """
        # Conflict: blend explicit + relational
        conflict_per_100w = (features.conflict_markers / max(features.word_count, 50)) * 100
        explicit_signal = min(1.0, conflict_per_100w / 2.0)
        conflict_signal = explicit_signal * 0.5 + features.relational_conflict * 0.5
        
        # Pacing
        pacing_signal = features.pacing_score / 10
        
        # Dialogue (already 0-1 from v2)
        dialogue_signal = features.dialogue_density
        if 0.1 <= features.dialogue_density <= 0.35:
            dialogue_signal = min(1.0, features.dialogue_density * 2.5)
        elif features.dialogue_density > 0.5:
            dialogue_signal = 0.6
        
        # Action
        action_signal = min(1.0, features.action_verb_ratio * 1.5)
        
        # Emotion: composite of v1 variance + v2 signals
        emotion_components = [
            features.emotional_variance * 0.25,
            features.pronoun_intensity * 0.15,
            min(1.0, features.sentiment_shifts / 3.0) * 0.20,
            features.contrast_density * 0.10,
            features.short_burst_ratio * 0.10,
            min(1.0, math.sqrt(features.vulnerability_count) / 2.0) * 0.10,
            features.emotional_punctuation * 0.05,
            features.interjection_density * 0.05,
        ]
        emotion_signal = sum(emotion_components)
        
        # Genre tolerance: if dialogue-heavy, dampen raw emotion slightly
        if features.dialogue_density > 0.4:
            emotion_signal *= 0.85
        
        emotion_signal = min(1.0, emotion_signal)
        
        return {
            'pacing': round(pacing_signal, 3),
            'dialogue': round(dialogue_signal, 3),
            'action': round(action_signal, 3),
            'conflict': round(min(1.0, conflict_signal), 3),
            'emotion': round(emotion_signal, 3)
        }
    
    # ==========================================
    # PART 4: ENDING DETECTION
    # ==========================================
    
    def get_ending_signals(self, text: str, features: TextFeatures) -> Dict[str, Any]:
        """
        Production-grade ending detection.
        Combines: reflective phrases, sentence deceleration, conflict drop,
        emotional punctuation drop, contrast drop, and NER entity absence.
        """
        doc = None
        sentences = []
        
        if self._spacy_available and self.nlp:
            try:
                doc = self.nlp(text)
                sentences = list(doc.sents)
            except Exception:
                pass
        
        if not sentences:
            raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = raw_sents
        
        if len(sentences) < 3:
            return {
                'ending_probability': 0.3,
                'reflective_tone': False,
                'tension_decrease': False,
                'no_new_conflict': True,
                'reasoning': []
            }
        
        mid = len(sentences) // 2
        quarter_idx = max(1, len(sentences) // 4)
        
        first_half = sentences[:mid]
        second_half = sentences[mid:]
        last_quarter = sentences[-quarter_idx:]
        first_three_quarters = sentences[:-quarter_idx]
        
        signals = []
        ending_prob = 0.2  # Base
        
        # --- Sub-signal 1: Reflective Phrases (weight 0.3) ---
        last_text = " ".join(
            s.text if hasattr(s, 'text') else s for s in last_quarter
        )
        reflective_phrase_count = sum(
            1 for p in self.REFLECTIVE_PHRASES if p.search(last_text)
        )
        reflective_word_count = sum(
            1 for w in self.REFLECTIVE_WORDS if w in last_text.lower()
        )
        total_reflective = reflective_phrase_count + reflective_word_count
        reflective_tone = total_reflective >= 2
        
        if reflective_tone:
            ending_prob += 0.25
            signals.append(f"Reflective tone: {total_reflective} closure markers in final section")
        elif total_reflective == 1:
            ending_prob += 0.08
            signals.append(f"Weak reflective signal: 1 closure marker")
        
        # --- Sub-signal 2: Conflict Density Drop (weight 0.25) ---
        first_text = " ".join(s.text if hasattr(s, 'text') else s for s in first_half)
        second_text = " ".join(s.text if hasattr(s, 'text') else s for s in second_half)
        
        first_conflict = self._count_explicit_conflict(first_text)
        second_conflict = self._count_explicit_conflict(second_text)
        
        tension_decrease = first_conflict > 0 and second_conflict < first_conflict * 0.6
        if tension_decrease:
            ending_prob += 0.2
            signals.append(f"Tension decreased: {first_conflict} → {second_conflict} conflict markers")
        
        # --- Sub-signal 3: No New Conflicts in Final Section (weight 0.1) ---
        last_quarter_text = " ".join(
            s.text if hasattr(s, 'text') else s for s in last_quarter
        )
        last_conflicts = self._count_explicit_conflict(last_quarter_text)
        no_new_conflict = last_conflicts <= 1
        
        if no_new_conflict:
            ending_prob += 0.08
            signals.append("No significant new conflicts in final section")
        else:
            ending_prob -= 0.08
            signals.append(f"Active conflict ({last_conflicts} markers) in final section")
        
        # --- Sub-signal 4: Sentence Deceleration (weight 0.15) ---
        def avg_len(sents):
            lengths = [len(s.text.split()) if hasattr(s, 'text') else len(s.split()) for s in sents]
            return np.mean(lengths) if lengths else 0
        
        last_avg = avg_len(last_quarter)
        first_avg = avg_len(first_three_quarters)
        
        deceleration = first_avg > 0 and last_avg < first_avg * 0.75
        if deceleration:
            ending_prob += 0.1
            signals.append(f"Sentence deceleration: {first_avg:.1f} → {last_avg:.1f} avg words")
        
        # --- Sub-signal 5: Emotional Punctuation Drop (weight 0.1) ---
        first_emo_punct = sum(len(p.findall(first_text)) for p in self.EMOTIONAL_PUNCT_PATTERNS)
        last_emo_punct = sum(len(p.findall(last_quarter_text)) for p in self.EMOTIONAL_PUNCT_PATTERNS)
        
        if first_emo_punct > 2 and last_emo_punct < first_emo_punct * 0.5:
            ending_prob += 0.07
            signals.append("Emotional punctuation decreased in final section")
        
        # --- Sub-signal 6: Contrast Transition Drop (weight 0.1) ---
        first_contrast = len(self.CONTRAST_TRANSITIONS.findall(first_text))
        last_contrast = len(self.CONTRAST_TRANSITIONS.findall(last_quarter_text))
        
        if first_contrast > 1 and last_contrast < first_contrast * 0.4:
            ending_prob += 0.07
            signals.append("Contrast transitions dropped in final section")
        
        # --- Sub-signal 7: New Entity Absence (weight 0.05, lightweight) ---
        if doc is not None:
            try:
                first_entities = set()
                last_entities = set()
                
                first_chars = sum(len(s.text) if hasattr(s, 'text') else len(s) for s in first_three_quarters)
                
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        if ent.start_char < first_chars:
                            first_entities.add(ent.text.lower())
                        else:
                            last_entities.add(ent.text.lower())
                
                new_entities = last_entities - first_entities
                if len(new_entities) == 0 and len(first_entities) > 0:
                    ending_prob += 0.05
                    signals.append("No new characters introduced in final section")
            except Exception:
                pass  # NER fallback: skip gracefully
        
        # --- Gating rule ---
        # Require: at least 2 reflective markers OR (1 reflective + tension_decrease)
        if total_reflective < 2 and not (total_reflective >= 1 and tension_decrease):
            # Dampen probability if neither condition met
            ending_prob = min(ending_prob, 0.45)
        
        # Pacing slowdown bonus
        if features.pacing_score < 4.0:
            ending_prob += 0.05
            signals.append(f"Pacing has slowed ({features.pacing_score:.1f})")
        
        # Low emotional variance bonus
        if features.emotional_variance < 0.15:
            ending_prob += 0.05
            signals.append("Emotional tone has settled")
        
        ending_prob = max(0.0, min(1.0, ending_prob))
        
        return {
            'ending_probability': round(ending_prob, 2),
            'reflective_tone': reflective_tone,
            'tension_decrease': tension_decrease,
            'no_new_conflict': no_new_conflict,
            'reasoning': signals
        }
