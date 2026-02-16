"""
Decision Engine - Production-Grade Narrative Evaluation
Enforces strict logical consistency between ending probability and continuation action.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum


class Decision(str, Enum):
    TRUE = "INTERESTING"
    FALSE = "LOW_ENGAGEMENT"
    AMBIGUOUS = "AMBIGUOUS"
    ENDING = "ENDING"
    CONTINUING = "CONTINUING"
    CONTINUE = "CONTINUE"
    CONCLUDE = "CONCLUDE"


@dataclass
class EngagementResult:
    is_interesting: Decision
    score: float
    confidence: float
    explanation: str
    result: str


@dataclass
class EndingResult:
    is_ending: Decision
    result: str
    probability: float
    confidence: float
    reasoning: str


@dataclass
class ContinuationResult:
    action: str
    reasoning: str
    confidence: float
    considerations: List[str]


class DecisionEngine:
    """
    Production-grade ensemble engine.
    
    CRITICAL RULES (from spec):
    1. ending.probability >= 0.6 -> action MUST be CONCLUDE
    2. ending.probability <= 0.4 -> action MUST be CONTINUE
    3. ending.probability 0.4-0.6 -> action may be AMBIGUOUS
    4. Engagement scores capped at 8.5 unless strong tension + emotional stakes
    5. Reasoning must be story-specific, never generic
    """
    
    HEURISTIC_WEIGHT = 0.35
    LLM_WEIGHT = 0.65
    
    ENGAGEMENT_THRESHOLD = 4.5
    ENDING_THRESHOLD = 0.6   # Updated: was 0.65
    
    # Engagement cap per spec
    ENGAGEMENT_CAP = 8.5
    
    def full_analysis(
        self,
        h_engagement: Dict[str, float],
        h_ending: Dict[str, float],
        l_engagement: Optional[Dict[str, Any]],
        l_ending: Optional[Dict[str, Any]],
        l_continuation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        eng_obj = self.decide_engagement(h_engagement, l_engagement)
        end_obj = self.decide_ending(h_ending, l_ending)
        cont_obj = self.decide_continuation(l_continuation)
        
        # Enforce logical consistency between ending prob and recommendation
        final_recommendation = self._enforce_consistency(
            end_obj.probability,
            eng_obj.score,
            cont_obj,
            h_engagement
        )
        
        return {
            'interesting': asdict(eng_obj),
            'ending': asdict(end_obj),
            'recommendation': asdict(final_recommendation)
        }
    
    def _enforce_consistency(
        self, 
        ending_prob: float, 
        engagement_score: float, 
        original_rec: ContinuationResult,
        h_eng: Dict[str, float]
    ) -> ContinuationResult:
        """
        STRICT consistency rules per production spec:
        
        1. ending_prob >= 0.6  -> MUST CONCLUDE
        2. ending_prob <= 0.4  -> MUST CONTINUE
        3. 0.4 < ending_prob < 0.6 -> AMBIGUOUS (use engagement as tiebreaker)
        
        Reasoning must reference actual signal values, never generic text.
        """
        
        pacing = h_eng.get('pacing', 0)
        conflict = h_eng.get('conflict', 0)
        emotion = h_eng.get('emotion', 0)
        
        # --- ZONE 1: CONCLUDE (ending_prob >= 0.6) ---
        if ending_prob >= 0.6:
            reasons = []
            if conflict < 0.2:
                reasons.append("conflict has subsided")
            if pacing < 0.3:
                reasons.append("pacing has slowed to reflective")
            if emotion < 0.2:
                reasons.append("emotional intensity has settled")
            if not reasons:
                reasons.append("narrative arc indicators suggest closure")
            
            return ContinuationResult(
                action="CONCLUDE",
                reasoning=f"Ending probability {ending_prob:.2f} exceeds threshold. {'; '.join(reasons).capitalize()}.",
                confidence=min(0.95, 0.7 + ending_prob * 0.3),
                considerations=["High ending probability", "Narrative closure detected"]
            )

        # --- ZONE 2: CONTINUE (ending_prob <= 0.4) ---
        if ending_prob <= 0.4:
            reasons = []
            if conflict > 0.3:
                reasons.append(f"active conflict signals ({conflict:.2f})")
            if pacing > 0.4:
                reasons.append(f"dynamic pacing ({pacing:.2f})")
            if emotion > 0.3:
                reasons.append(f"emotional variance ({emotion:.2f}) suggests unresolved arcs")
            if not reasons:
                reasons.append("open narrative threads remain")
            
            return ContinuationResult(
                action="CONTINUE",
                reasoning=f"Ending probability {ending_prob:.2f} is below threshold. {'; '.join(reasons).capitalize()}.",
                confidence=min(0.95, 0.7 + (1 - ending_prob) * 0.3),
                considerations=["Low ending probability", "Active narrative threads"]
            )
            
        # --- ZONE 3: AMBIGUOUS (0.4 < ending_prob < 0.6) ---
        # Use engagement as a tiebreaker
        if engagement_score >= 7.0:
            action = "CONTINUE"
            tie_reason = f"High engagement ({engagement_score:.1f}) favors maintaining momentum"
        elif engagement_score <= 4.0:
            action = "CONCLUDE"
            tie_reason = f"Low engagement ({engagement_score:.1f}) suggests wrapping up"
        else:
            action = "AMBIGUOUS"
            tie_reason = f"Engagement ({engagement_score:.1f}) is neutral"
        
        return ContinuationResult(
            action=action,
            reasoning=f"Ambiguous zone (ending prob {ending_prob:.2f}). {tie_reason}. Conflict={conflict:.2f}, Pacing={pacing:.2f}.",
            confidence=0.65,
            considerations=["Ambiguous ending signals", "Engagement used as tiebreaker"]
        )
    
    def decide_engagement(self, h_eng, l_eng) -> EngagementResult:
        """
        Engagement scoring with production-grade rules:
        - Cap at 8.5 unless strong tension + emotional stakes + narrative drive
        - Slice-of-life stories fall 5.5-7.5
        - Story-specific reasoning required
        """
        signal_weights = {
            'pacing': 0.25,
            'conflict': 0.30,
            'emotion': 0.20,
            'action': 0.15,
            'dialogue': 0.10
        }
        
        weighted_avg = sum(h_eng.get(key, 0) * weight for key, weight in signal_weights.items())
        active_signals = sum(1 for key in signal_weights if h_eng.get(key, 0) > 0.15)
        
        h_score = 3.0 + (weighted_avg * 14.0)
        
        if active_signals >= 4:
            h_score += 0.5
        elif active_signals >= 3:
            h_score += 0.3
        
        # PRODUCTION RULE: Cap at 8.5 unless exceptional
        tension_strong = h_eng.get('conflict', 0) > 0.5
        emotion_strong = h_eng.get('emotion', 0) > 0.4
        pacing_strong = h_eng.get('pacing', 0) > 0.5
        
        exceptional = tension_strong and emotion_strong and pacing_strong
        
        if not exceptional:
            h_score = min(self.ENGAGEMENT_CAP, h_score)
        
        h_score = max(0.0, min(10.0, h_score))
        
        # Build story-specific reasoning from actual signals
        reasoning_parts = []
        pacing_val = h_eng.get('pacing', 0)
        conflict_val = h_eng.get('conflict', 0)
        emotion_val = h_eng.get('emotion', 0)
        dialogue_val = h_eng.get('dialogue', 0)
        action_val = h_eng.get('action', 0)
        
        if conflict_val > 0.4:
            reasoning_parts.append(f"Strong conflict presence ({conflict_val:.2f}) drives tension")
        elif conflict_val > 0.15:
            reasoning_parts.append(f"Moderate conflict ({conflict_val:.2f}) maintains reader interest")
        else:
            reasoning_parts.append(f"Low conflict ({conflict_val:.2f}) — narrative relies on other elements")
            
        if pacing_val > 0.5:
            reasoning_parts.append(f"Dynamic pacing ({pacing_val:.2f}) with varied sentence rhythm")
        elif pacing_val < 0.2:
            reasoning_parts.append(f"Steady, measured pacing ({pacing_val:.2f})")
            
        if emotion_val > 0.3:
            reasoning_parts.append(f"Notable emotional variance ({emotion_val:.2f}) suggests shifting tone")
        
        if dialogue_val > 0.3:
            reasoning_parts.append(f"Dialogue-heavy ({dialogue_val:.2f}) which energizes the narrative")
        
        explanation = ". ".join(reasoning_parts) + "."
        
        if not l_eng or not l_eng.get('success'):
            res = Decision.TRUE if h_score > self.ENGAGEMENT_THRESHOLD else Decision.FALSE
            return EngagementResult(
                is_interesting=res,
                result=res.value,
                score=round(h_score, 1),
                confidence=0.65,
                explanation=explanation
            )
        
        l_data = l_eng['data']
        l_score = l_data['score']
        
        # Also cap LLM score
        if not exceptional:
            l_score = min(self.ENGAGEMENT_CAP, l_score)
        
        combined_score = (h_score * self.HEURISTIC_WEIGHT) + (l_score * self.LLM_WEIGHT)
        
        if not exceptional:
            combined_score = min(self.ENGAGEMENT_CAP, combined_score)
        
        agreement = 1.0 - (abs(h_score - l_score) / 10)
        confidence = (l_data['confidence'] * 0.7) + (agreement * 0.3)
        
        if abs(h_score - l_score) > 3.0:
            return EngagementResult(
                is_interesting=Decision.AMBIGUOUS,
                result=Decision.AMBIGUOUS.value,
                score=round(combined_score, 1),
                confidence=round(confidence * 0.7, 2),
                explanation=f"Signal disagreement: heuristic={h_score:.1f}, semantic={l_score:.1f}. {explanation}"
            )
        
        res = Decision.TRUE if combined_score > self.ENGAGEMENT_THRESHOLD else Decision.FALSE
        return EngagementResult(
            is_interesting=res,
            result=res.value,
            score=round(combined_score, 1),
            confidence=round(confidence, 2),
            explanation=explanation
        )
    
    def decide_ending(
        self, 
        h_end: Dict[str, float], 
        l_end: Optional[Dict[str, Any]]
    ) -> EndingResult:
        """
        Ending detection with literary closure awareness.
        Evaluates: conflict resolution, tonal shift, tension decrease.
        """
        h_prob = h_end.get('ending_probability', 0.3)
        conflict_level = h_end.get('conflict_level', 0)
        pacing_level = h_end.get('pacing_level', 0)
        
        # Build story-specific ending reasoning
        ending_signals = []
        continuing_signals = []
        
        if h_prob > 0.5:
            if conflict_level < 3:
                ending_signals.append("minimal remaining conflict markers")
            if pacing_level < 4:
                ending_signals.append("pacing has slowed, suggesting reflective closure")
        else:
            if conflict_level >= 3:
                continuing_signals.append(f"{conflict_level} active conflict markers")
            if pacing_level >= 5:
                continuing_signals.append(f"pacing score {pacing_level:.1f} indicates ongoing momentum")
        
        if not l_end or not l_end.get('success'):
            res = Decision.ENDING if h_prob > self.ENDING_THRESHOLD else Decision.CONTINUING
            
            if ending_signals:
                reasoning = f"Heuristic ending probability {h_prob:.2f}. Signals: {'; '.join(ending_signals)}."
            elif continuing_signals:
                reasoning = f"Heuristic ending probability {h_prob:.2f}. Continuation signals: {'; '.join(continuing_signals)}."
            else:
                reasoning = f"Heuristic ending probability {h_prob:.2f}. Mixed signals from conflict and pacing analysis."
            
            return EndingResult(
                is_ending=res,
                result=res.value,
                probability=round(h_prob, 2),
                confidence=0.55,
                reasoning=reasoning
            )
        
        l_data = l_end['data']
        l_prob = l_data['probability']
        combined_prob = (h_prob * self.HEURISTIC_WEIGHT) + (l_prob * self.LLM_WEIGHT)
        
        res = Decision.ENDING if combined_prob > self.ENDING_THRESHOLD else Decision.CONTINUING
        
        # Combine heuristic insights with LLM data for richer reasoning
        if ending_signals:
            reasoning = f"Ending probability {combined_prob:.2f}. {'; '.join(ending_signals).capitalize()}."
        elif continuing_signals:
            reasoning = f"Ending probability {combined_prob:.2f}. {'; '.join(continuing_signals).capitalize()}."
        else:
            reasoning = f"Ending probability {combined_prob:.2f}. Balanced signals — no strong closure or continuation indicators."
        
        return EndingResult(
            is_ending=res,
            result=res.value,
            probability=round(combined_prob, 2),
            confidence=round(l_data['confidence'], 2),
            reasoning=reasoning
        )
    
    def decide_continuation(
        self, 
        l_cont: Optional[Dict[str, Any]]
    ) -> ContinuationResult:
        if not l_cont or not l_cont.get('success'):
            return ContinuationResult(
                action=Decision.AMBIGUOUS.value,
                reasoning="Insufficient semantic data for continuation assessment.",
                confidence=0.0,
                considerations=[]
            )
        
        l_data = l_cont['data']
        return ContinuationResult(
            action=l_data['action'],
            reasoning=l_data['reasoning'],
            confidence=round(l_data['confidence'], 2),
            considerations=l_data.get('considerations', [])
        )
    
    def generate_author_insights(
        self,
        eng_result: EngagementResult,
        end_result: EndingResult, 
        cont_result: ContinuationResult,
        h_signals: Dict[str, float]
    ) -> str:
        """
        Generate author-friendly insights without numbers.
        Returns a narrative paragraph with actionable guidance.
        """
        insights = []
        
        # Engagement insights (qualitative)
        if eng_result.score >= 8.0:
            insights.append("Your narrative demonstrates strong reader engagement through compelling conflict and emotional resonance. The pacing keeps readers invested, with clear stakes that matter. This is the kind of writing that makes readers eager to turn the page.")
        elif eng_result.score >= 6.5:
            insights.append("Your story maintains solid momentum with good emotional beats and character development. There are engaging elements present, though deepening the conflict or raising the stakes could amplify reader investment even further. Consider where tension might naturally escalate.")
        elif eng_result.score >= 5.0:
            insights.append("Your narrative has foundational elements in place, but readers may need stronger hooks to stay fully engaged. Look for opportunities to increase dramatic tension, clarify what's at stake for your characters, or tighten the pacing in slower sections. Even subtle conflicts can create compelling reading.")
        else:
            insights.append("This section may benefit from stronger narrative drive. Consider introducing clear conflict, raising emotional stakes, or varying your pacing to create more dynamic reader engagement. Strong stories often balance quieter moments with tension that propels readers forward.")
        
        # Specific craft insights based on signals
        craft_notes = []
        conflict = h_signals.get('conflict', 0)
        emotion = h_signals.get('emotion', 0)
        pacing = h_signals.get('pacing', 0)
        dialogue = h_signals.get('dialogue', 0)
        
        if conflict < 0.2:
            craft_notes.append("introducing tension or obstacles that challenge your characters")
        if emotion < 0.2:
            craft_notes.append("deepening emotional stakes so readers connect with what matters to your characters")
        if pacing < 0.3:
            craft_notes.append("varying sentence rhythm and scene structure to create forward momentum")
        if dialogue < 0.15 and conflict > 0.3:
            craft_notes.append("using dialogue to reveal character and advance conflict naturally")
        
        if craft_notes:
            insights.append(f"To strengthen this passage, consider {', '.join(craft_notes[:-1]) + (' and ' + craft_notes[-1] if len(craft_notes) > 1 else craft_notes[0])}.")
        
        # Ending/continuation guidance
        if end_result.is_ending == Decision.ENDING:
            if cont_result.action == "CONCLUDE":
                insights.append("Your narrative threads are converging toward resolution. The emotional and plot arcs appear ready for closure. If this is your intended ending, ensure you've addressed the core questions and tensions you've established. A satisfying conclusion resonates with the journey you've taken readers on.")
            else:
                insights.append("While some closure elements are present, you may want to ensure all significant story threads reach satisfying resolution before concluding. Readers will appreciate seeing how conflicts and character arcs complete their trajectories.")
        else:
            if cont_result.action == "CONTINUE":
                insights.append("Your story has narrative momentum that suggests more to explore. The conflicts remain unresolved and there's potential for deeper character development. Readers are likely curious about where the journey leads next, which is an excellent position for continuing the narrative.")
            elif cont_result.action == "AMBIGUOUS":
                insights.append("You're at a pivotal point where the narrative could either continue or conclude effectively. Consider what serves your story best: does the emotional arc feel complete, or are there compelling threads worth exploring further? Trust your storytelling instincts about what this narrative needs.")
            else:
                insights.append("Though some plot elements remain open, the narrative energy may benefit from resolution rather than extension. Sometimes the most impactful stories know when to conclude, leaving readers satisfied rather than overstaying the emotional moment.")
        
        return " ".join(insights)