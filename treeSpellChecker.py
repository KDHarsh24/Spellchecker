import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from trieDataStructure import Trie

# Initialize and load Trie
trie = Trie()
trie.load_from_file()

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower().replace("  ", " ")


# Function to correct text using the Trie
def correct_text(src_file, tgt_file):
    corrected_lines = []
    try:
        # Count total lines for tqdm progress bar
        with open(src_file, "r", encoding="utf-8") as file:
            total_lines = sum(1 for _ in file)

        with open(src_file, "r", encoding="utf-8") as file, open(tgt_file, "w", encoding="utf-8") as output_file:
            for line in tqdm(file, total=total_lines, desc="🔍 Correcting Text", unit="line"):
                words = clean_text(line).split()  # Tokenize words
                corrected_words = []
                for word in words:
                    levenD = 2
                    correct_words = []

                    # Gradually increase Levenshtein distance until a match is found
                    while not correct_words and levenD < len(word):
                        correct_words = trie.find_similar_words(word, max_distance=levenD)
                        levenD += 1

                    corrected_words.append(correct_words[0] if correct_words else word)

                corrected_line = " ".join(corrected_words)
                corrected_lines.append(corrected_line)
                output_file.write(corrected_line + "\n")

        print(f"\n✅ Corrected output saved to '{tgt_file}'")

    except FileNotFoundError:
        print(f"\n❌ Error: File '{src_file}' not found!")

    return corrected_lines


# Function to compare generated output with expected output and calculate BLEU
def compare_outputs(predicted_file, ground_truth_file):
    mismatched_lines = []
    smoothing_fn = SmoothingFunction().method4

    try:
        with open(predicted_file, "r", encoding="utf-8") as pred_file, open(ground_truth_file, "r", encoding="utf-8") as true_file:
            pred_lines_raw = pred_file.readlines()
            true_lines_raw = true_file.readlines()

            total_lines = len(true_lines_raw)
            pred_lines = [clean_text(line).split() for line in pred_lines_raw]
            true_lines = [clean_text(line).split() for line in true_lines_raw]

            correct_count = 0
            line_bleu_scores = []

            for idx in tqdm(range(total_lines), desc="🔎 Comparing Outputs", unit="line"):
                pred = pred_lines[idx]
                true = true_lines[idx]

                if pred == true:
                    correct_count += 1
                else:
                    mismatched_lines.append(
                        f"Line {idx + 1}:\nPredicted: {' '.join(pred)}\nExpected:  {' '.join(true)}\n"
                    )

                # BLEU per line
                line_bleu = sentence_bleu([true], pred, smoothing_function=smoothing_fn)
                line_bleu_scores.append(line_bleu)

            # Corpus BLEU Score
            corpus_score = corpus_bleu([[true] for true in true_lines], pred_lines, smoothing_function=smoothing_fn)

            accuracy = (correct_count / total_lines) * 100 if total_lines > 0 else 0

            print(f"\n📊 Accuracy: {accuracy:.2f}% ({correct_count}/{total_lines} lines matched)")
            print(f"📈 Average Line BLEU Score: {sum(line_bleu_scores)/len(line_bleu_scores):.4f}")
            print(f"🏆 Corpus BLEU Score: {corpus_score:.4f}")

            # Save mismatched lines to file
            if mismatched_lines:
                with open("mismatched_lines.txt", "w", encoding="utf-8") as mismatch_file:
                    mismatch_file.writelines("\n".join(mismatched_lines))
                print("❌ Mismatched lines saved to 'mismatched_lines.txt'")

            return accuracy, corpus_score, line_bleu_scores, mismatched_lines

    except FileNotFoundError:
        print("❌ Error: One of the files is missing!")
        return 0, 0, [], []


# Run the correction and comparison pipeline
corrected_output_file = "generated_output.train.tgt"
corrected_lines = correct_text("artificial.train.src", corrected_output_file)
accuracy, corpus_bleu, line_bleus, mismatches = compare_outputs(corrected_output_file, "artificial.train.tgt")


# # Show top 10 mismatches if available
# if mismatches:
#     print("\n⚠️ Mismatched Lines (Top 10):")
#     print("\n".join(mismatches[:10]))  # First 10 mismatches
