import re
from typing import Dict, List, Any, Optional

class TaxonomyEngine:
    """
    Deterministic inference engine for mapping stories to internal taxonomy.
    Strictly rule-based: No LLMs, No RNG.
    """
    
    TAXONOMY = {
        "Fiction": {
            "Romance": ["Slow-burn", "Enemies-to-Lovers", "Second Chance"],
            "Thriller": ["Espionage", "Psychological", "Legal Thriller"],
            "Sci-Fi": ["Hard Sci-Fi", "Space Opera", "Cyberpunk"],
            "Horror": ["Psychological Horror", "Gothic", "Slasher"]
        }
    }
    
    # Weighted Keywords for Deterministic Scoring - EXPANDED WITH SYNONYMS
    # Structure: Subgenre -> List of (keyword, weight)
    # Weights: 1.0 = standard, 2.0 = strong signal, 0.5 = weak signal
    KEYWORDS = {
        # Romance
        "Slow-burn": [
            ("slow burn", 2.0), ("lingering", 1.0), ("gradual", 1.0), ("patience", 0.5),
            ("unspoken", 1.0), ("tension", 1.2), ("smolder", 1.5), ("yearning", 1.3),
            ("falls in love", 1.8), ("falling for", 1.5), ("love", 1.2)
        ],
        "Enemies-to-Lovers": [
            ("enemies", 2.0), ("hate", 1.5), ("rival", 1.5), ("conflict", 0.5), 
            ("bicker", 1.0), ("argument", 0.8), ("antagonist", 1.3),
            ("adversar", 1.5), ("nemesis", 1.8), ("clash", 1.0), ("friction", 1.2),
            ("falls in love", 1.8), ("falling for", 1.5), ("love", 1.2)
        ],
        "Second Chance": [
            ("second chance", 2.0), ("reunite", 1.5), ("ex", 1.0), ("past", 0.5), 
            ("again", 0.5), ("reconcil", 1.8), ("forgiv", 1.3), ("reunion", 1.5),
            ("return", 0.8), ("rekindle", 1.7), ("falls in love", 1.8),
            ("falling for", 1.5), ("love", 1.2)
        ],
        
        # Thriller
        "Espionage": [
            ("spy", 2.0), ("agent", 2.0), ("covert", 1.5), ("mission", 1.0), 
            ("intelligence", 1.5), ("secret", 1.0), ("kremlin", 1.5), ("cia", 1.5),
            ("operative", 1.8), ("asset", 1.3), ("handler", 1.5), ("mole", 1.8),
            ("surveillance", 1.5), ("classified", 1.3), ("espionage", 2.0), ("mi6", 1.5)
        ],
        "Psychological": [
            ("mind", 1.5), ("game", 1.0), ("manipulat", 1.5), ("gaslight", 2.0), 
            ("sanity", 1.5), ("psycho", 1.8), ("mental", 1.2), ("paranoi", 1.7),
            ("obsess", 1.5), ("control", 1.0), ("twist", 1.3), ("perception", 1.4)
        ],
        "Legal Thriller": [
            ("legal", 2.0), ("court", 2.0), ("lawyer", 2.0), ("attorney", 2.0), 
            ("judge", 1.5), ("verdict", 1.5), ("trial", 1.5), ("jury", 1.0),
            ("case", 1.0), ("counsel", 1.5), ("litigation", 1.8), ("defense", 1.2),
            ("prosecutor", 1.7), ("witness", 1.3), ("evidence", 1.2), ("testimony", 1.3)
        ],
        
        # Sci-Fi
        "Hard Sci-Fi": [
            ("physics", 2.0), ("quantum", 1.5), ("scientific", 1.5), ("accuracy", 1.0),
            ("relativity", 1.8), ("equation", 1.5), ("theory", 1.2), ("research", 1.0),
            ("experiment", 1.3), ("professor", 0.8), ("scientist", 1.2), ("technical", 1.3)
        ],
        "Space Opera": [
            ("space", 1.5), ("ship", 1.5), ("galaxy", 2.0), ("empire", 1.5), 
            ("star", 1.0), ("war", 0.3), ("fleet", 1.7), ("planet", 1.2),
            ("captain", 1.0), ("admiral", 1.3), ("interstellar", 1.8), ("spacecraft", 1.6),
            ("alien", 1.2), ("sector", 1.0), ("federation", 1.5), ("starship", 1.6)
        ],
        "Cyberpunk": [
            ("cyber", 2.0), ("neon", 1.5), ("hacker", 1.5), ("corp", 1.0), 
            ("tech", 1.0), ("implant", 1.5), ("ai", 1.2), ("tokyo", 1.0), ("future", 1.0),
            ("augment", 1.5), ("virtual", 1.3), ("matrix", 1.7), ("network", 1.0),
            ("megacorp", 1.8), ("dystopi", 1.5), ("chrome", 1.2), ("jack in", 1.8),
            ("data", 1.0), ("neural", 1.5), ("cybernetic", 1.8)
        ],
        
        # Horror
        "Psychological Horror": [
            ("madness", 1.5), ("hallucinat", 1.5), ("dread", 1.5), ("mind", 1.0),
            ("insanity", 1.7), ("delusion", 1.6), ("terror", 1.3), ("unease", 1.4),
            ("anxiety", 1.2), ("disturb", 1.4), ("nightmare", 1.3), ("fear", 1.0)
        ],
        "Gothic": [
            ("gothic", 2.0), ("mansion", 2.0), ("curse", 1.5), ("family secret", 1.5), 
            ("ghost", 1.0), ("haunt", 1.5), ("victorian", 1.5), ("corridor", 1.0), 
            ("whisper", 1.0), ("estate", 1.8), ("decay", 1.5), ("shadow", 1.2),
            ("ancient", 1.3), ("castle", 1.7), ("portrait", 1.2), ("ancestor", 1.3),
            ("candle", 1.0), ("fog", 1.2), ("mist", 1.2), ("ruin", 1.4)
        ],
        "Slasher": [
            ("killer", 2.0), ("knife", 1.5), ("mask", 1.5), ("teen", 1.0), 
            ("camp", 1.0), ("blood", 1.0), ("murder", 1.8), ("victim", 1.3),
            ("stalk", 1.5), ("scream", 1.2), ("chase", 1.3), ("blade", 1.4),
            ("massacre", 1.7), ("slaughter", 1.6)
        ]
    }
    
    # Thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.12  # Minimum confidence to be mapped (score >= 0.5)
    AMBIGUITY_MARGIN = 0.35         # If top 2 scores are within 35%, flag as ambiguous
    MULTI_LABEL_THRESHOLD = 0.50    # Secondary matches above this are included
    
    # Contextual Boosting: Keyword combos that strongly signal a subgenre
    COMBO_BONUSES = {
        "Cyberpunk": [
            (["neon", "cyber"], 1.5),
            (["neon", "corp"], 1.3),
            (["hacker", "corp"], 1.4),
            (["augment", "neural"], 1.6),
            (["virtual", "matrix"], 1.5)
        ],
        "Gothic": [
            (["mansion", "curse"], 1.5),
            (["victorian", "ghost"], 1.4),
            (["ancient", "shadow"], 1.3),
            (["decay", "estate"], 1.4)
        ],
        "Espionage": [
            (["spy", "mission"], 1.4),
            (["agent", "covert"], 1.5),
            (["intelligence", "asset"], 1.6)
        ],
        "Space Opera": [
            (["galaxy", "empire"], 1.6),
            (["fleet", "war"], 1.4),
            (["star", "ship"], 1.2)
        ]
    }
    
    # Weights for input source
    WEIGHT_BLURB = 1.0   # Context Wins!
    WEIGHT_TAGS = 0.4    # Tags are supplementary/noisy
    
    # Non-fiction indicators - if detected, reject classification
    NON_FICTION_INDICATORS = [
        "how to", "guide", "tutorial", "instructions", "learn", "step by step",
        "diy", "beginner", "tips", "tricks", "advice", "handbook", "manual",
        "introduction to", "basics of", "getting started", "cookbook", "recipe"
    ]
    
    def infer(self, tags_input: str, blurb_input: str, multi_label: bool = True) -> Dict[str, Any]:
        """
        Main inference method with multi-label support.
        Returns mapped category/categories with confidence and reasoning.
        
        Args:
            tags_input: Comma-separated tags
            blurb_input: Story description/blurb
            multi_label: If True, returns top N matches above threshold
        """
        # 1. Preprocess
        tags_list = [t.strip().lower() for t in tags_input.split(',') if t.strip()]
        blurb_norm = blurb_input.lower()
        
        # Check for non-fiction indicators
        for indicator in self.NON_FICTION_INDICATORS:
            if indicator in blurb_norm or any(indicator in tag for tag in tags_list):
                return {
                    "status": "UNMAPPED",
                    "reason": f"Non-fiction content detected ('{indicator}'). This taxonomy only handles Fiction.",
                    "category": None, "genre": None, "subgenre": None,
                    "confidence": 0.0, "details": {},
                    "all_matches": []
                }
        
        scores = {}  # (Genre, Subgenre) -> score
        reasons = {} # (Genre, Subgenre) -> list of triggers
        matched_keywords = {}  # Track which keywords matched for combo detection
        
        # 2. Scoring Loop
        for genre, subgenres in self.TAXONOMY["Fiction"].items():
            for subgenre in subgenres:
                key = (genre, subgenre)
                if subgenre not in self.KEYWORDS:
                    continue
                
                score = 0.0
                triggered_by = []
                keywords_found = []
                
                # Check keywords
                for keyword, weight in self.KEYWORDS[subgenre]:
                    # Check Blurb (Higher Weight)
                    pattern = r'\b' + re.escape(keyword) + r'\w*\b' # greedy match
                    matches = re.findall(pattern, blurb_norm)
                    if matches:
                        points = len(matches) * weight * self.WEIGHT_BLURB
                        score += points
                        triggered_by.append(f"Blurb: '{keyword}' ({points:.2f})")
                        keywords_found.append(keyword)
                    
                    # Check Tags (Lower Weight)
                    for tag in tags_list:
                         if keyword in tag: # substring match is fine for tags
                            points = weight * self.WEIGHT_TAGS
                            score += points
                            triggered_by.append(f"Tag: '{tag}' matched '{keyword}' ({points:.2f})")
                            keywords_found.append(keyword)
                
                # 3. Apply Contextual Boosting
                if subgenre in self.COMBO_BONUSES:
                    for combo_keywords, bonus in self.COMBO_BONUSES[subgenre]:
                        if all(kw in keywords_found for kw in combo_keywords):
                            score += bonus
                            triggered_by.append(f"Combo bonus: {'+'.join(combo_keywords)} (+{bonus:.2f})")
                
                if score > 0:
                    scores[key] = score
                    reasons[key] = triggered_by
                    matched_keywords[key] = keywords_found

        # 4. Decision Logic
        if not scores:
            return {
                "status": "UNMAPPED",
                "reason": "No taxonomy keywords found in input.",
                "category": None, "genre": None, "subgenre": None,
                "confidence": 0.0, "details": {},
                "all_matches": []
            }
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_match = sorted_scores[0]
        top_key = top_match[0]
        top_score = top_match[1]
        
        # Calculate confidence (normalized)
        confidence = min(1.0, top_score / 4.0)
        
        # 5. Multi-label Detection
        all_matches = []
        for (genre, subgenre), score in sorted_scores:
            match_confidence = min(1.0, score / 4.0)
            if match_confidence >= self.MULTI_LABEL_THRESHOLD or (genre, subgenre) == top_key:
                all_matches.append({
                    "genre": genre,
                    "subgenre": subgenre,
                    "confidence": round(match_confidence, 2),
                    "score": round(score, 2),
                    "triggers": reasons[(genre, subgenre)][:3]
                })
        
        # 6. Ambiguity Detection
        is_ambiguous = False
        ambiguity_reason = ""
        
        if len(sorted_scores) > 1:
            second_match = sorted_scores[1]
            second_score = second_match[1]
            
            # Check margin
            if top_score > 0 and (top_score - second_score) / top_score < self.AMBIGUITY_MARGIN:
                is_ambiguous = True
                ambiguity_reason = f"Close match with {second_match[0][1]} (Score: {second_score:.2f})"
        
        # 7. Threshold Check
        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
             return {
                "status": "UNMAPPED",
                "reason": f"Low confidence ({confidence:.2f}). Best match was {top_key[1]} but score was too low.",
                "category": None, "genre": None, "subgenre": None,
                "confidence": confidence,
                "details": {f"{k[0]} > {k[1]}": round(v, 2) for k, v in scores.items()},
                "all_matches": all_matches if multi_label else []
            }
            
        # 8. Format Output
        # Ambiguity takes priority over multi-label when scores are close
        status = "AMBIGUOUS" if is_ambiguous else ("MULTI_LABEL" if (multi_label and len(all_matches) > 1) else "MAPPED")
        
        return {
            "status": status,
            "category": "Fiction",
            "genre": top_key[0],
            "subgenre": top_key[1],
            "confidence": round(confidence, 2),
            "reasoning": f"Matched {len(reasons[top_key])} signals. Top: {', '.join(reasons[top_key][:3])}. {ambiguity_reason}",
            "details": {
                "all_scores": {f"{k[0]} > {k[1]}": round(v, 2) for k, v in scores.items()},
                "triggers": reasons[top_key]
            },
            "all_matches": all_matches if multi_label else [all_matches[0]] if all_matches else []
        }