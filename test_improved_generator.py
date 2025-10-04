#!/usr/bin/env python3
"""
Test script to demonstrate the improved maze generator with Veronica Boyce's methodology
"""

from maze_generator import MazeGenerator

def test_improved_generator():
    """Test the improved generator to show that repeated words are fixed"""
    print("Testing Improved Maze Generator with Veronica Boyce's Methodology")
    print("=" * 70)
    
    try:
        # Initialize the generator
        generator = MazeGenerator()
        
        # Test sentences
        test_sentences = [
            "Joseph brewed the beer he will serve next week, but it is not very tasty.",
            "The producer replaced the actor, and the actress quit the movie after the fight.",
            "Kim will display the photos she took last month, but she won't show all of them."
        ]
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\nTest {i}: {sentence}")
            print("-" * 50)
            
            # Reset word tracker for each sentence
            generator.reset_global_word_tracker()
            
            # Generate maze pairs
            result = generator.generate_full_maze(sentence)
            
            # Display results
            for pair in result["maze_pairs"]:
                print(f"[Position {pair['position']}] {pair['correct']} vs {pair['distractor']}")
            
            # Check for repeated words
            distractors = [pair['distractor'] for pair in result["maze_pairs"] if pair['distractor'] != 'x-x-x']
            unique_distractors = set(distractors)
            repeated_count = len(distractors) - len(unique_distractors)
            
            print(f"\nDistractor Analysis:")
            print(f"- Total distractors: {len(distractors)}")
            print(f"- Unique distractors: {len(unique_distractors)}")
            print(f"- Repeated words: {repeated_count}")
            
            if repeated_count == 0:
                print("✅ SUCCESS: No repeated words found!")
            else:
                print("❌ ISSUE: Found repeated words")
                repeated_words = []
                for word in distractors:
                    if distractors.count(word) > 1 and word not in repeated_words:
                        repeated_words.append(word)
                print(f"Repeated words: {repeated_words}")
            
            print("=" * 70)
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_generator()

