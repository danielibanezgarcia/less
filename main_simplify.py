from config import horn_simplifier

if __name__ == "__main__":
    lex_mturk_file = 'data/horn/lex.mturk.es.txt'
    candidates_file = 'candidates.txt'
    horn_simplifier.simplify(lex_mturk_file, candidates_file)
