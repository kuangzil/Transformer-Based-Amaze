#!/usr/bin/env python3
"""
Test script to verify the API-only maze generator without hardcoded fallback words
"""

from maze_generator import MazeGenerator
import time

def test_api_only_generator():
    """Test the API-only generator to ensure no hardcoded words are used"""
    print("Testing API-Only Maze Generator")
    print("=" * 50)
    
    try:
        # Initialize the generator
        generator = MazeGenerator()
        
        # Test sentence
        test_sentence = "Joseph brewed the beer he will serve next week, but it is not very tasty."
        print(f"Test sentence: {test_sentence}")
        print("-" * 50)
        
        # Reset word tracker
        generator.reset_global_word_tracker()
        
        print("Generating maze pairs (this may take a moment due to API calls)...")
        start_time = time.time()
        
        # Generate maze pairs
        result = generator.generate_full_maze(test_sentence)
        
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        print("-" * 50)
        
        # Display results
        print("Generated maze pairs:")
        for pair in result["maze_pairs"]:
            print(f"[Position {pair['position']}] {pair['correct']} vs {pair['distractor']}")
        
        # Analyze results
        distractors = [pair['distractor'] for pair in result["maze_pairs"] if pair['distractor'] != 'x-x-x']
        unique_distractors = set(distractors)
        repeated_count = len(distractors) - len(unique_distractors)
        
        print(f"\nAnalysis:")
        print(f"- Total distractors: {len(distractors)}")
        print(f"- Unique distractors: {len(unique_distractors)}")
        print(f"- Repeated words: {repeated_count}")
        print(f"- All distractors: {distractors}")
        
        # Check for hardcoded words (these should NOT appear)
        hardcoded_words = [
            "elephant", "piano", "guitar", "bicycle", "car", "house", "tree",
            "water", "fire", "wind", "earth", "sky", "big", "small", "red",
            "blue", "green", "yellow", "black", "white", "fast", "slow"
        ]
        
        found_hardcoded = []
        for distractor in distractors:
            if distractor.lower() in hardcoded_words:
                found_hardcoded.append(distractor)
        
        if not found_hardcoded:
            print("✅ SUCCESS: No hardcoded fallback words found!")
            print("✅ The generator is using API-generated words only.")
        else:
            print(f"❌ ISSUE: Found hardcoded words: {found_hardcoded}")
        
        if repeated_count == 0:
            print("✅ SUCCESS: No repeated words found!")
        else:
            print("❌ ISSUE: Found repeated words")
            repeated_words = []
            for word in distractors:
                if distractors.count(word) > 1 and word not in repeated_words:
                    repeated_words.append(word)
            print(f"Repeated words: {repeated_words}")
        
        print("=" * 50)
        
        # Test fallback mechanism specifically
        print("\nTesting fallback mechanism...")
        print("Testing _random_corpus_word method directly...")
        
        # Reset for fallback test
        generator.used_words_in_sentence = set()
        
        fallback_words = []
        for i in range(5):
            word = generator._random_corpus_word()
            fallback_words.append(word)
            print(f"Fallback word {i+1}: {word}")
        
        # Check if any fallback words are hardcoded
        hardcoded_fallback = [w for w in fallback_words if w.lower() in hardcoded_words]
        if not hardcoded_fallback:
            print("✅ SUCCESS: Fallback mechanism uses API-generated words only!")
        else:
            print(f"❌ ISSUE: Fallback mechanism found hardcoded words: {hardcoded_fallback}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_only_generator()


