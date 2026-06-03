import torch
from pathlib import Path
import stanza
import math


# Uses GPU (if available), otherwise CPU
# https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use is: {device}")
    return device


# Loads dataset from a file
def load_data(file_path):
    file_path = Path(file_path)
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    data = [line.strip() for line in data if line.strip()] # Removes empty lines (to avoid parsing errors)
    print(f"Number of documents loaded: {len(data)}")
    return data


# Initializes the stanza pipeline
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

# Builds vocabularies & noun lists & verb lists from the entire dataset
def build_token_indexes(data):
    # Joins all lines into one text so Stanza can process the entire dataset at once
    text = " ".join(data)
    doc = nlp(text)
    vocabulary = []
    nouns = []
    verbs = []

    for sentence in doc.sentences:
        for token in sentence.words:
		    # In order to treat "A" and "a" the same
            token_text = token.text.lower()
            # In order to treat "tears" and "tearing" and "tear" as same verb
            lemma = token.lemma.lower()

            # Stores each unique token in the vocabulary
            if token_text not in vocabulary:
                vocabulary.append(token_text)

            # For unique NOUN lemmas
            if token.upos == "NOUN" and lemma not in nouns:
                nouns.append(lemma)

            # For unique VERB lemmas
            if token.upos == "VERB" and lemma not in verbs:
                verbs.append(lemma)

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Number of nouns: {len(nouns)}")
    print(f"Number of verbs: {len(verbs)}")

    return vocabulary, nouns, verbs



# Converts 1 parsed Stanza sentence into a vector
# [subject_noun_id, root_verb_id, object_noun_id]
def getEmbedding(sentence, nouns, verbs):
    vec = [0, 0, 0]

    for token in sentence.words:
        lemma = token.lemma.lower()

        if token.deprel == "nsubj":
            if lemma in nouns:
                vec[0] = nouns.index(lemma) + 1

        if token.deprel == "root":
            if lemma in verbs:
                vec[1] = verbs.index(lemma) + 1

        if token.deprel == "obj":
            if lemma in nouns:
                vec[2] = nouns.index(lemma) + 1

    # Calculates the vector length
    norm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    if norm > 0:
        return [vec[0] / norm, vec[1] / norm, vec[2] / norm]
    # Else
    return [0, 0, 0]



# Converts the dataset (all sentences) into vectors
def texts_to_vectors(data, nouns, verbs):
    vectors = {}
    # Parse the whole dataset again because we need Stanza sentence objects, not raw text
    text = " ".join(data)
    doc = nlp(text)

    for sentence in doc.sentences:
        vector = getEmbedding(sentence, nouns, verbs)
        vectors[tuple(vector)] = sentence.text

    return vectors


# The calls
device = get_device()
data = load_data(""".../file.xxx""")
vocabulary, nouns, verbs = build_token_indexes(data)
vectors = texts_to_vectors(data, nouns, verbs)
print(vectors)