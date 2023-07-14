from config import horn_candidates_file, horn_lex_mturk_file, horn_scorer, horn_scores_file

if __name__ == "__main__":

    # horn_simplifier.simplify(horn_lex_mturk_file, horn_candidates_file)
    with open(horn_scores_file, 'w') as results_file:
        for n_candidates in (1, 2, 3, 4, 5, 10):
            precision, accuracy, changed = horn_scorer.score(horn_lex_mturk_file, horn_candidates_file, n_candidates)
            results_file.write("\n")
            results_file.write(f"n-best:    {n_candidates}\n")
            results_file.write(f"Precision: {precision:.4f}\n")
            results_file.write(f"Accuracy:  {accuracy:.4f}\n")
            results_file.write(f"Changed:   {changed:.4f}\n")
