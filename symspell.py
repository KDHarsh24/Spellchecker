from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os


# -------------------- SymSpell Initialization --------------------
def initialize_symspell(dictionary_path, max_edit_distance=2, prefix_length=7):
    """
    Initialize SymSpell with a given dictionary.
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        print("Dictionary file not found or incorrect format.")
        exit(1)
    print("SymSpell dictionary loaded successfully.")
    return sym_spell


# -------------------- File Correction Function --------------------
def correct_file(input_file_path, output_file_path, sym_spell, compound=False):
    """
    Corrects spelling errors in a file using SymSpell and saves the output.
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if compound:
                # Compound correction for full sentences
                suggestions = sym_spell.lookup_compound(line.strip(), max_edit_distance=2)
                corrected_line = suggestions[0].term if suggestions else line.strip()
            else:
                # Word-by-word correction
                corrected_line = []
                words = line.strip().split()
                for word in words:
                    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                    corrected_word = suggestions[0].term if suggestions else word
                    corrected_line.append(corrected_word)
                corrected_line = ' '.join(corrected_line)
            
            outfile.write(corrected_line + '\n')
    print(f"Corrected file saved to: {output_file_path}")


# -------------------- BLEU Score and Accuracy Calculation --------------------
def calculate_bleu_and_accuracies(reference_file, hypothesis_file):
    """
    Calculates BLEU score, token-level accuracy, and line-level accuracy.
    """
    bleu_scores = []
    token_accuracies = []
    line_matches = 0
    total_lines = 0
    smoothie = SmoothingFunction().method4  # Smoothing for short sentences

    with open(reference_file, 'r', encoding='utf-8') as ref_file, \
         open(hypothesis_file, 'r', encoding='utf-8') as hyp_file:
        for idx, (ref_line, hyp_line) in enumerate(zip(ref_file, hyp_file), start=1):
            ref_line = ref_line.strip()
            hyp_line = hyp_line.strip()

            ref_tokens = ref_line.split()
            hyp_tokens = hyp_line.split()
            
            # BLEU score for the line
            bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie, weights=(0.5, 0.5))
            bleu_scores.append(bleu_score)
            
            # Token-level accuracy for the line
            correct_tokens = sum(1 for ref_token, hyp_token in zip(ref_tokens, hyp_tokens) if ref_token == hyp_token)
            total_tokens = max(len(ref_tokens), len(hyp_tokens))  # To account for insertion/deletion errors
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
            token_accuracies.append(token_accuracy)

            # Line match check (exact match)
            if ref_line == hyp_line:
                line_matches += 1

            total_lines += 1

            print(f"[Line {idx}] BLEU: {bleu_score:.4f}, Token Accuracy: {token_accuracy:.4f}, Line Match: {'✅' if ref_line == hyp_line else '❌'}")

    # Average scores
    avg_bleu = sum(bleu_scores) / total_lines if total_lines else 0
    avg_token_accuracy = sum(token_accuracies) / total_lines if total_lines else 0
    line_level_accuracy = line_matches / total_lines if total_lines else 0

    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    print(f"Average Token-Level Accuracy: {avg_token_accuracy:.4f}")
    print(f"Line-Level Accuracy (Exact Match): {line_level_accuracy:.4f} ({line_matches}/{total_lines} lines correct)")

    return avg_bleu, avg_token_accuracy, line_level_accuracy


# -------------------- Main Function --------------------
def main():
    # -------- File Paths --------
    dictionary_path = "dictionary.txt"  # Pre-downloaded frequency dictionary
    input_file_path = "artificial.train.src"              # Input file with spelling errors
    output_file_path = "corrected_output.txt"              # Output file after corrections
    reference_file_path = "artificial.train.tgt"               # Ground truth reference file

    # -------- File Check --------
    for path in [dictionary_path, input_file_path, reference_file_path]:
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            return

    # -------- Initialize SymSpell --------
    sym_spell = initialize_symspell(dictionary_path)

    # -------- Correct File --------
    correct_file(input_file_path, output_file_path, sym_spell, compound=False)  # set compound=True for sentence-level correction

    # -------- BLEU and Accuracy --------
    calculate_bleu_and_accuracies(reference_file_path, output_file_path)


# -------------------- Execute Script --------------------
if __name__ == "__main__":
    main()
