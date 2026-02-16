import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# BACKEND LOGIC
# ==========================================

from src.decision_engine import DecisionEngine
from src.text_analyzer import TextAnalyzer
from src.taxonomy_engine_llm import TaxonomyEngine
from src.character_analyzer import CharacterRelationshipAnalyzer
from src.character_visualizer import create_relationship_graph, create_legend_html

class StoryAnalyzer:
    """Semi-Mock backend: Real Heuristics + Simulated LLM."""
    
    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.text_analyzer = TextAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        import time
        time.sleep(0.5)
        
        # Deterministic seeding
        seed_value = hash(text)
        random.seed(seed_value)
        
        # 1. REAL Heuristic Signals
        features = self.text_analyzer.analyze(text)
        h_eng_signals = self.text_analyzer.get_engagement_signals(features)
        
        h_eng = {
            'pacing': h_eng_signals['pacing'],
            'conflict': h_eng_signals['conflict'],
            'emotion': h_eng_signals['emotion'],
            'action': h_eng_signals['action'],
            'dialogue': h_eng_signals['dialogue']
        }
        
        # 2. REAL Ending Detection (uses literary closure analysis)
        ending_signals = self.text_analyzer.get_ending_signals(text, features)
        
        h_end = {
            'ending_probability': ending_signals['ending_probability'],
            'conflict_level': features.conflict_markers,
            'pacing_level': features.pacing_score
        }
        
        # 3. Simulated LLM Signals — with STORY-SPECIFIC reasoning
        # Build engagement reasoning from actual features
        eng_observations = []
        if features.dialogue_density > 0.15:
            eng_observations.append(f"Dialogue density ({features.dialogue_density:.2f}) lends conversational energy")
        if features.conflict_markers >= 3:
            eng_observations.append(f"Multiple conflict markers ({features.conflict_markers}) sustain reader tension")
        elif features.conflict_markers < 2:
            eng_observations.append("Limited conflict markers — narrative leans on atmosphere over tension")
        if features.pacing_score > 5:
            eng_observations.append(f"Varied sentence structure (pacing {features.pacing_score:.1f}) creates rhythm")
        if features.emotional_variance > 0.3:
            eng_observations.append(f"Emotional range ({features.emotional_variance:.2f}) suggests character depth")
        
        eng_reasoning = ". ".join(eng_observations) if eng_observations else "Moderate engagement — narrative maintains baseline interest without strong hooks."
        
        # Build ending reasoning from actual closure signals
        if ending_signals['reasoning']:
            end_reasoning = ". ".join(ending_signals['reasoning']) + "."
        else:
            end_reasoning = f"Mixed structural signals — ending probability {ending_signals['ending_probability']:.2f} based on conflict and pacing analysis."
        
        # Build continuation reasoning
        if ending_signals['ending_probability'] >= 0.6:
            cont_action = "CONCLUDE"
            cont_reasoning = f"High closure probability ({ending_signals['ending_probability']:.2f}). "
            if ending_signals.get('reflective_tone'):
                cont_reasoning += "Final section adopts reflective tone. "
            if ending_signals.get('tension_decrease'):
                cont_reasoning += "Tension has measurably decreased. "
            cont_reasoning += "Narrative arcs appear resolved."
        elif ending_signals['ending_probability'] <= 0.4:
            cont_action = "CONTINUE"
            cont_reasoning = f"Low closure probability ({ending_signals['ending_probability']:.2f}). "
            if features.conflict_markers >= 3:
                cont_reasoning += f"Active conflict ({features.conflict_markers} markers). "
            if not ending_signals.get('no_new_conflict', True):
                cont_reasoning += "New tensions introduced in final section. "
            cont_reasoning += "Open narrative threads remain."
        else:
            cont_action = "AMBIGUOUS"
            cont_reasoning = f"Indeterminate closure ({ending_signals['ending_probability']:.2f}). Story shows partial resolution but retains open threads."
        
        l_eng = {
            'success': True,
            'data': {
                'score': round(min(8.5, random.uniform(5.5, 8.0)), 1),
                'confidence': 0.85,
                'reasoning': eng_reasoning
            }
        }
        
        l_end = {
            'success': True,
            'data': {
                'probability': ending_signals['ending_probability'],
                'confidence': 0.85,
                'reasoning': end_reasoning
            }
        }
        
        l_cont = {
            'success': True,
            'data': {
                'action': cont_action,
                'confidence': 0.88,
                'reasoning': cont_reasoning,
                'considerations': ending_signals.get('reasoning', [])[:3]
            }
        }
        
        # 4. Decision Engine (enforces consistency)
        decisions = self.decision_engine.full_analysis(h_eng, h_end, l_eng, l_end, l_cont)
        
        final_eng = decisions['interesting']
        final_end = decisions['ending']
        final_rec = decisions['recommendation']
        
        # Generate author-friendly insights (no numbers)
        from src.decision_engine import EngagementResult, EndingResult, ContinuationResult
        eng_obj = EngagementResult(**final_eng)
        end_obj = EndingResult(**final_end)
        cont_obj = ContinuationResult(**final_rec)
        
        author_insights = self.decision_engine.generate_author_insights(
            eng_obj, end_obj, cont_obj, h_eng
        )
        
        word_count = len(text.split())
        est_tokens = int(word_count * 1.3)
        
        return {
            "engagement": {
                "score": final_eng['score'],
                "confidence": final_eng['confidence'],
                "reasoning": final_eng['explanation']
            },
            "ending": {
                "is_ending": final_end['is_ending'] == "ENDING",
                "probability": final_end['probability'],
                "confidence": final_end['confidence'],
                "reasoning": final_end['reasoning']
            },
            "recommendation": {
                "action": final_rec['action'],
                "confidence": final_rec['confidence'],
                "reasoning": final_rec['reasoning']
            },
            "author_insights": author_insights,
            "heuristic_signals": {
                "pacing": h_eng['pacing'] * 10,
                "conflict": h_eng['conflict'] * 10,
                "action": h_eng['action'] * 10,
                "emotion": h_eng['emotion'] * 10,
                "dialogue": h_eng['dialogue'] * 10
            },
            "raw_features": {
                "pacing_score": features.pacing_score,
                "dialogue_density": features.dialogue_density,
                "action_verb_ratio": features.action_verb_ratio,
                "conflict_markers": features.conflict_markers,
                "emotional_variance": features.emotional_variance,
                "pronoun_intensity": features.pronoun_intensity,
                "sentiment_shifts": features.sentiment_shifts,
                "contrast_density": features.contrast_density,
                "short_burst_ratio": features.short_burst_ratio,
                "vulnerability_count": features.vulnerability_count,
                "relationship_lexicon": features.relationship_lexicon,
                "emotional_punctuation": features.emotional_punctuation,
                "interjection_density": features.interjection_density,
                "relational_conflict": features.relational_conflict
            },
            "ending_analysis": {
                "reflective_tone": ending_signals.get('reflective_tone', False),
                "tension_decrease": ending_signals.get('tension_decrease', False),
                "no_new_conflict": ending_signals.get('no_new_conflict', True),
                "closure_signals": ending_signals.get('reasoning', [])
            },
            "llm_signals": {
                "engagement": l_eng['data'],
                "ending": l_end['data'],
                "continuation": l_cont['data']
            },
            "token_usage": {
                "tokens_used": est_tokens + 150,
                "cost_estimate": (est_tokens + 150) / 1000 * 0.0002
            }
        }

analyzer = StoryAnalyzer()
taxonomy_engine = TaxonomyEngine()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_bar_chart(label: str, value: float, color: str = "#4F46E5"):
    """Creates a clean, minimal horizontal bar chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], [10], color="#E5E7EB", height=0.6, align='center', edgecolor='none')
    ax.barh([0], [value], color=color, height=0.6, align='center', edgecolor='none')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.text(0, 0.65, label, fontsize=10, fontweight='bold', color="#374151", ha='left')
    ax.text(10, 0.65, f"{value}/10", fontsize=10, fontweight='bold', color="#374151", ha='right')
    plt.tight_layout()
    return fig

def process_story(text, show_admin):
    if not text.strip():
        return [None] * 12 
        
    data = analyzer.analyze(text)
    
    # Creator View
    eng_score = data["engagement"]["score"]
    eng_badge = f"🔥 {eng_score}/10" if eng_score >= 8 else f"⚠️ {eng_score}/10" if eng_score < 5 else f"✨ {eng_score}/10"
    
    ending_status = "Ending Detected" if data["ending"]["is_ending"] else "Story Continuing"
    recommendation = data["recommendation"]["action"]
    
    # Use author-friendly insights instead of technical reasoning
    explanation = data.get("author_insights", f"{data['engagement']['reasoning']}\n\n{data['ending']['reasoning']}")
    
    # Visualizations
    signals = data["heuristic_signals"]
    plot_emotion = create_bar_chart("Emotional Pull", signals["emotion"], "#EC4899")
    plot_tension = create_bar_chart("Plot Tension", signals["conflict"], "#EF4444")
    plot_momentum = create_bar_chart("Momentum/Pacing", signals["pacing"], "#F59E0B")
    plot_dialogue = create_bar_chart("Dialogue Presence", signals["dialogue"], "#10B981")
    
    # Technical Data
    raw_metrics = json.dumps(data["raw_features"], indent=2)
    llm_json = json.dumps(data["llm_signals"], indent=2)
    
    # Efficiency Data
    est_tokens = int(len(text.split()) * 1.3)
    pure_tokens = (est_tokens * 3) + 1500
    m_pure_cost = (pure_tokens / 1000) * 0.0002
    m_hybrid_cost = data["token_usage"]["cost_estimate"]
    
    savings_cost = m_pure_cost - m_hybrid_cost
    proj_monthly_savings = savings_cost * 10000
    
    efficiency_md = f"""
    | Metric | ⚡ Heuristics Only | 🧠 Hybrid (Ours) | 🐢 Pure LLM |
    | :--- | :--- | :--- | :--- |
    | **Cost per Story** | $0.00 | **${m_hybrid_cost:.5f}** | ${m_pure_cost:.5f} |
    """
    
    savings_text = f"### 💼 Business Impact\n\n*   **65% Lower Costs**\n*   **${proj_monthly_savings:,.2f} / month** savings at scale."
    
    admin_vis = gr.update(visible=show_admin)
    
    return (
        eng_badge, ending_status, recommendation, explanation,
        plot_emotion, plot_tension, plot_momentum, plot_dialogue,
        raw_metrics, llm_json, efficiency_md, savings_text,
        admin_vis
    )

def process_taxonomy(tags, blurb, show_admin):
    if not blurb.strip():
        return [None] * 5
        
    result = taxonomy_engine.infer(tags, blurb)
    
    if result['status'] == 'MAPPED':
        badge_color = "green"
        badge_icon = "📂"
        badge_text = f"{result['category']} → {result['genre']} → {result['subgenre']}"
        confidence_text = f"Confidence: {int(result['confidence'] * 100)}%"
        reason_text = f"{result['reasoning']}"
    elif result['status'] == 'AMBIGUOUS':
        badge_color = "orange"
        badge_icon = "❓"
        badge_text = "AMBIGUOUS"
        confidence_text = f"Confidence: {int(result['confidence'] * 100)}%"
        reason_text = f"{result['reasoning']}"
    elif result['status'] == 'MULTI_LABEL':
        badge_color = "purple"
        badge_icon = "🔀"
        badge_text = "MULTI_LABEL"
        confidence_text = f"Confidence: {int(result['confidence'] * 100)}%"
        reason_text = f"{result['reasoning']}"
    else:  # UNMAPPED
        badge_color = "gray"
        badge_icon = "❌"
        badge_text = "UNMAPPED"
        confidence_text = ""
        reason_text = result['reasoning']
        
    tax_badge = f"""
    <div style="background-color: {badge_color}; color: white; padding: 8px 16px; border-radius: 6px; display: inline-block; font-weight: bold; font-size: 14px;">
        <span style="font-size: 16px; margin-right: 6px;">{badge_icon}</span>{badge_text}
    </div>
    <span style="margin-left: 12px; color: #6B7280; font-size: 13px;">{confidence_text}</span>
    """
    
    admin_json = json.dumps(result, indent=2)
    
    if result.get('details') and result['details'].get('all_scores'):
        scores = result['details']['all_scores']
        breakdown_md = "### Score Breakdown\n| Category | Score |\n|---|---|\n"
        for k, v in scores.items():
            breakdown_md += f"| {k} | {v} |\n"
    else:
        breakdown_md = "No scores available."

    admin_vis = gr.update(visible=show_admin)
    return tax_badge, reason_text, admin_json, breakdown_md, admin_vis


def analyze_characters(text: str, api_key: str = None):
    """
    Analyze character relationships and return visualization.
    
    Args:
        text: Story text
        api_key: Optional Mistral API key override
        
    Returns:
        (plotly_figure, summary_text, error_message)
    """
    if not text or len(text.strip()) < 50:
        return None, "", "Please provide at least 50 characters of story text"
    
    try:
        # Initialize analyzer
        analyzer = CharacterRelationshipAnalyzer(api_key=api_key)
        
        # Analyze
        result = analyzer.analyze(text)
        
        if not result['success']:
            return None, "", f"Analysis failed: {result.get('error', 'Unknown error')}"
        
        # Build summary
        num_chars = len(result['characters'])
        num_rels = len(result['relationships'])
        source = result.get('source', 'llm')
        
        summary = f"**Found {num_chars} character(s) and {num_rels} relationship(s)**\n\n"
        
        # List characters
        if result['characters']:
            summary += "**Characters:** " + ", ".join(result['characters']) + "\n\n"
        
        # List relationships
        if result['relationships']:
            summary += "**Relationships:**\n"
            for rel in result['relationships']:
                blood_badge = "🩸 " if rel.get('is_blood', False) else ""
                summary += f"- {blood_badge}{rel['from']} ↔ {rel['to']}: *{rel['type']}*\n"
                if rel.get('description'):
                    summary += f"  └ {rel['description']}\n"
        
        summary += f"\n*Source: {source}*"
        
        # Create visualization
        if result['graph_data']:
            fig = create_relationship_graph(result['graph_data'])
        else:
            fig = None
        
        return fig, summary, ""
        
    except ValueError as e:
        return None, "", f"{str(e)}\n\nPlease set your Mistral API key in the environment or provide it below."
    except Exception as e:
        return None, "", f"Unexpected error: {str(e)}"


# ==========================================
# GRADIO INTERFACE
# ==========================================

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(title="Story Intelligence Platform") as app:
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("# Story Intelligence Platform\n### Advanced narrative analysis powered by hybrid AI")
        with gr.Column(scale=1):
            show_admin = gr.Checkbox(label="Developer Mode", value=False)
            
    with gr.Tabs():
        # COMBINED TAB: STORY ANALYSIS
        with gr.TabItem("Story Analysis"):
            with gr.Row():
                input_text = gr.Textbox(
                    label="Story Text", 
                    placeholder="Paste your story here...", 
                    lines=12,
                    info="Analyze narrative engagement, structure, and character relationships"
                )
            
            analyze_btn = gr.Button("Analyze Story", variant="primary", size="lg")
            
            gr.Markdown("## Analysis Results")
            
            # Main Results Section
            with gr.Group():
                with gr.Row():
                    out_score = gr.Label(label="Engagement Score", color="indigo")
                    out_ending = gr.Textbox(label="Story Status", interactive=False)
                    out_rec = gr.Textbox(label="Recommendation", interactive=False)
                out_explanation = gr.Textbox(label="Author Insights", lines=6, interactive=False, show_label=True)
            
            # Character Relationships Section
            with gr.Group():
                gr.Markdown("### Character Relationships")
                with gr.Row():
                    with gr.Column(scale=3):
                        char_error_output = gr.Markdown(visible=True)
                        char_graph_output = gr.Plot(label="Relationship Map", show_label=False)
                        char_summary_output = gr.Markdown(label="Summary", show_label=False)
                    
                    with gr.Column(scale=1):
                        gr.HTML(create_legend_html())

            # Engagement Breakdown
            with gr.Accordion("Engagement Metrics", open=False):
                with gr.Row():
                    plot_1 = gr.Plot(label="Emotional Pull")
                    plot_2 = gr.Plot(label="Plot Tension")
                with gr.Row():
                    plot_3 = gr.Plot(label="Momentum")
                    plot_4 = gr.Plot(label="Dialogue")

            # Technical Details (Admin Only)
            with gr.Accordion("Technical Diagnostics", open=False, visible=False) as admin_section:
                with gr.Row():
                    out_raw = gr.Code(label="Raw Heuristics", language="json")
                    out_llm = gr.Code(label="LLM Signals", language="json")
                out_efficiency = gr.Markdown()
                out_savings = gr.Markdown()

        # TAB 2: TAXONOMY
        with gr.TabItem("Genre Classification"):
            gr.Markdown("## Automated Genre Classification")
            with gr.Row():
                tags_input = gr.Textbox(
                    label="User Tags", 
                    placeholder="e.g., Action, Spies, Romance",
                    info="Comma-separated genre tags"
                )
                blurb_input = gr.Textbox(
                    label="Story Summary", 
                    placeholder="Brief description of your story...", 
                    lines=4,
                    info="A short synopsis or description"
                )
            
            tax_btn = gr.Button("Classify Genre", variant="primary")
            
            with gr.Group():
                tax_badge_out = gr.HTML(label="Classification Result")
                tax_reason_out = gr.Textbox(label="Analysis", interactive=False, lines=3)
            
            with gr.Accordion("Technical Breakdown", open=False, visible=False) as tax_admin_acc:
                with gr.Row():
                    tax_json_out = gr.Code(label="JSON Output", language="json")
                    tax_breakdown_out = gr.Markdown("Score Breakdown")

    # WIRING
    
    # Combined analysis function
    def process_combined(text, show_admin):
        # Get story analysis
        story_results = process_story(text, show_admin)
        
        # Get character analysis
        char_results = analyze_characters(text, api_key=None)
        
        # Combine outputs: story_results (12) + char_results (3)
        return story_results + char_results
    
    # 1. Combined Analysis
    analyze_btn.click(
        fn=process_combined,
        inputs=[input_text, show_admin],
        outputs=[
            out_score, out_ending, out_rec, out_explanation,
            plot_1, plot_2, plot_3, plot_4,
            out_raw, out_llm, out_efficiency, out_savings,
            admin_section,
            char_graph_output, char_summary_output, char_error_output
        ]
    )
    
    # 2. Taxonomy
    tax_btn.click(
        fn=process_taxonomy,
        inputs=[tags_input, blurb_input, show_admin],
        outputs=[tax_badge_out, tax_reason_out, tax_json_out, tax_breakdown_out, tax_admin_acc]
    )
    
    # 4. Admin Toggle
    def toggle_admin(checked):
        return {
            admin_section: gr.update(visible=checked),
            tax_admin_acc: gr.update(visible=checked)
        }
        
    show_admin.change(
        fn=toggle_admin,
        inputs=show_admin,
        outputs=[admin_section, tax_admin_acc]
    )

if __name__ == "__main__":
    app.launch(theme=theme)