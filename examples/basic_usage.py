"""
Example usage of the Narrative Intelligence System.
"""

from story_analyzer import NarrativeIntelligence


def main():
    # Sample story for analysis
    story = """
    Sarah was a brilliant detective in the bustling city of New York.
    She had solved countless mysteries throughout her career.
    One dark evening, a mysterious case landed on her desk.
    "This is strange," Sarah muttered as she examined the clues.
    
    Her partner, Detective James, joined her in the investigation.
    "What do you think happened here?" James asked nervously.
    Sarah studied the evidence carefully, her mind racing.
    
    The case involved a missing artifact from the museum.
    Strange symbols and cryptic messages were left at the scene.
    Sarah felt both fear and excitement as she pursued the truth.
    
    Days turned into weeks as they chased leads across the city.
    Danger lurked around every corner, but Sarah never gave up.
    Finally, after a dramatic chase through the subway tunnels,
    they cornered the thief and recovered the precious artifact.
    
    "We did it!" James exclaimed with joy and relief.
    Sarah smiled, feeling victorious and accomplished.
    Justice had been served, and the city was safe once more.
    """

    # Initialize the Narrative Intelligence System
    print("=== Narrative Intelligence System Demo ===\n")
    nis = NarrativeIntelligence()

    # Perform comprehensive analysis
    print("Analyzing story...")
    results = nis.analyze(story)

    # Display results
    print(f"\n--- Basic Info ---")
    print(f"Text length: {results['text_length']} characters")
    print(f"Word count: {results['word_count']} words")
    print(f"Analysis mode: {results['analysis_mode']}")

    # Engagement analysis
    print(f"\n--- Engagement Analysis ---")
    engagement = results['engagement']
    print(f"Overall engagement score: {engagement['engagement_score']:.3f}")
    print(f"Pacing variance: {engagement['pacing_variance']:.3f}")
    print(f"Dialogue density: {engagement['dialogue_density']:.3f}")
    print(f"Action density: {engagement['action_density']:.3f}")
    print(f"Emotional intensity: {engagement['emotional_intensity']:.3f}")
    print(f"Hook quality: {engagement['hook_quality']:.3f}")
    print(f"Readability: {engagement['readability']:.3f}")

    # Genre classification
    print(f"\n--- Genre Classification ---")
    genre = results['genre']
    print(f"Primary genre: {genre['primary_genre']}")
    print(f"Is mixed genre: {genre['is_mixed_genre']}")
    print(f"Tone: {genre['tone']}")
    print("Genre scores:")
    for g, score in genre['genre_scores'].items():
        print(f"  {g}: {score:.3f}")
    if genre['subgenres']:
        print(f"Subgenres: {', '.join(genre['subgenres'])}")

    # Ending prediction
    print(f"\n--- Ending Prediction ---")
    ending = results['ending_prediction']
    print(f"Predicted ending type: {ending['predicted_ending']}")
    print(f"Confidence: {ending['confidence']:.3f}")
    print(f"Arc completion: {ending['arc_completion']:.3f}")
    print(f"Resolution likelihood: {ending['resolution_likelihood']:.3f}")
    print(f"Emotional trajectory: {ending['emotional_trajectory']}")

    # Character analysis
    print(f"\n--- Character Analysis ---")
    characters = results['characters']
    print(f"Total characters found: {len(characters['characters'])}")
    print(f"Characters: {', '.join(characters['characters'])}")
    if characters['main_characters']:
        print(f"Main characters: {', '.join(characters['main_characters'])}")
    print(f"Network density: {characters['network_density']:.3f}")
    print(f"\nTop character relationships:")
    for rel in characters['relationships'][:5]:  # Show top 5
        print(f"  {rel['character1']} <-> {rel['character2']} (strength: {rel['strength']})")

    # Summary
    print(f"\n--- Summary ---")
    summary = results['summary']
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Example: Analyze individual components
    print("\n\n=== Individual Component Examples ===\n")

    # Engagement only
    print("--- Engagement Analysis Only ---")
    engagement_only = nis.analyze_engagement(story)
    print(f"Engagement score: {engagement_only['engagement_score']:.3f}")

    # Genre only
    print("\n--- Genre Classification Only ---")
    genre_only = nis.classify_genre(story)
    print(f"Primary genre: {genre_only['primary_genre']}")

    # Ending prediction only
    print("\n--- Ending Prediction Only ---")
    ending_only = nis.predict_ending(story)
    print(f"Predicted ending: {ending_only['predicted_ending']}")

    # Character network
    print("\n--- Character Network Data ---")
    network = nis.get_character_network(story)
    print(f"Nodes: {len(network['nodes'])}")
    print(f"Edges: {len(network['edges'])}")


if __name__ == "__main__":
    main()
