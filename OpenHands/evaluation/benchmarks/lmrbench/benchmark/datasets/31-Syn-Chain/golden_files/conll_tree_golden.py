import spacy

def spacy_result_to_conll(sentence):
    """
    Generates a dependency parse for the input sentence in the CoNLL-U format.

    Args:
        sentence (`str`): An input sentence.
    
    Retruns:
        conll_output (`str`): A dependency parse for the input sentence in the CoNLL-U format.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    header = ["ID", "text", "LEMMA", "POS", "Tag", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
    conll_output = "{:<5} {:<10} {:<10} {:<6} {:<6} {:<6} {:<6} {:<8} {:<6} {:<6}\n".format(*header)

    for token in doc:
        row = [
            token.i + 1,  # ID
            token.text,  # FORM
            token.lemma_,  # LEMMA
            token.pos_,  # UPOS
            token.tag_,  # XPOS
            "_",  # FEATS
            token.head.i + 1,  # HEAD
            token.dep_,  # DEPREL
            "_",  # DEPS
            "_"  # MISC
        ]
        conll_output += "{:<5} {:<10} {:<10} {:<6} {:<6} {:<6} {:<6} {:<8} {:<6} {:<6}\n".format(*row)

    return conll_output


