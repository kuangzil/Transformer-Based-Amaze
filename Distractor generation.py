import pandas as pd
import json
from maze_generator import MazeGenerator  # original MazeGenerator class

def apply_maze_generator(sentence):
    generator = MazeGenerator()
    print(f"\nProcessing sentence: {sentence}")
    pairs = []
    for pair in generator.generate_full_maze_stream(sentence):  # generate pairs one by one
        print(f"[Position {pair['position']}] {pair['correct']} vs {pair['distractor']}")
        pairs.append(pair)
    return json.dumps(pairs, ensure_ascii=False)




if __name__ == "__main__":

    input_csv = r"C:\Users\likua\OneDrive\Desktop\juliet Emaze\AMAZE_GPT\boyce_data\selected_trial_01.csv" # input CSV path
    output_csv = r"C:\Users\likua\OneDrive\Desktop\juliet Emaze\AMAZE_GPT\boyce_data\test_material_01.csv" # save path

    df = pd.read_csv(input_csv)

    # assuming there is a column called "sentence"
    df["maze_pairs"] = df["sentence"].apply(apply_maze_generator)

    # saving the DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print("Maze generation completed and saved to", output_csv)
