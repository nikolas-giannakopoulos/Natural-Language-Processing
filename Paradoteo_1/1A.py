def tokenize(sentence):
    tokens = []
    word = ""
 
    for character in sentence:

        # Έλεγχος αν ο χαρακτήρας είναι αλφαριθμητικός
        if character.isalnum():
            word += character
        else:

            # Αν δεν είναι αλφαριθμητικός, σημαίνει ότι τελείωσε μια λέξη ή σημεία στίξης/διάστημα
            # Αν η μεταβλητή 'word' δεν είναι κενή, σημαίνει ότι υπάρχει μια λέξη για προσθήκη
            if word:
                tokens.append(word.lower())
                word = ""

            if character.strip():  # Non-whitespace
                tokens.append(character)

    # Εάν υπάρχει, προσθήκη τελευταίας λέξης
    if word:
        tokens.append(word.lower())

    return tokens

dictionary = {
    "thank": "V",         # Ρήμα
    "your": "PRON",       # Αντωνυμία
    "message": "N",       # Ουσιαστικό
    "to": "PREP",         # Πρόθεση
    "show": "V",          # Ρήμα
    "our": "PRON",        # Αντωνυμία
    "words": "N",         # Ουσιαστικό
    "the": "DET",         # Άρθρο
    "doctor": "N",        # Ουσιαστικό
    "as": "PREP",         # Πρόθεση
    "his": "PRON",        # Αντωνυμία
    "next": "ADJ",        # Επίθετο
    "contract": "N",      # Ουσιαστικό
    "checking": "V",      # Ρήμα
    "all": "DET",         # Προσδιοριστής
    "of": "PREP",         # Πρόθεση
    "us": "PRON",         # Αντωνυμία
    ".": "PUNCT",         # Σημείο στίξης
    "we": "PRON",         # Αντωνυμία
    "should": "V",        # Ρήμα
    "be": "V",            # Ρήμα
    "grateful": "ADJ",    # Επίθετο
    "i": "PRON",          # Αντωνυμία
    "mean": "V",          # Ρήμα
    "for": "PREP",        # Πρόθεση
    "acceptance": "N",    # Ουσιαστικό
    "and": "CONJ",        # Και
    "efforts": "N",       # Ουσιαστικό
    "until": "PREP",      # Πρόθεση
    "springer": "N",      # Ουσιαστικό 
    "link": "N",          # Ουσιαστικό
    "came": "V",          # Ρήμα
    "finally": "ADV",     # Επίρρημα
    "last": "ADJ",        # Επίθετο
    "week": "N",          # Ουσιαστικό
    "think": "V",         # Ρήμα
    ",": "PUNCT",         # Σημείο στίξης
}

# Όλα τα ρήματα
verbs = {word for word, tag in dictionary.items() if tag == "V"}

def tag(tokens):
 
    # Αντιστοίχιση κάθε token στην αντίστοιχη ετικέτα POS από το λεξικό, εαν δεν, ετικέτα "UNK"
    match = [(token, dictionary.get(token, "UNK")) for token in tokens]
 
    for i in range(len(match) - 1):

        if match[i][0] == "to":
            match[i] = ("to", "TO" if match[i+1][1] == "V" else "PREP")

    return match

# Ορισμός ενός συνόλου κανόνων μετασχηματισμού
# Κάθε κανόνας είναι μια tuple (pattern, replacement)
rules = [
    # Κανόνας 1
    # Εισαγωγή του "you" μετά το "thank"
    (
        [("thank", "V")],
        [("thank", "V"), ("you", "PRON")]
    ),

    # Κανόνας 2
    # Εισαγωγή του "for" πριν από το "your message"
    (
        [("you", "PRON"), ("your", "PRON"), ("message", "N")],
        [("you", "PRON"), ("for", "PREP"), ("your", "PRON"), ("message", "N")]
    ),

    # Κανόνας 3
    # Αντικατάσταση του "as" με "during"
    (
        [("as", "PREP")],
        [("during", "PREP")]
    ),

    # Κανόνας 4
    # Αντικατάσταση του "checking" με "review"
    (
        [("checking", "V")],
        [("review", "N")]
    )
]

def apply_rules(tagged):

    new_tagged = list(tagged)

    for pattern, replacement in rules:
        pattern_len = len(pattern)
        i = 0
        while i <= len(new_tagged) - pattern_len:
            if new_tagged[i:i+pattern_len] == pattern:

                # Αντικατάσταση του pattern με το replacement στη λίστα
                new_tagged = new_tagged[:i] + replacement + new_tagged[i+pattern_len:]
                
                # Προσαρμογή του δείκτη 'i' για να συνεχίσει η σάρωση μετά την αντικατάσταση
                i += len(replacement) - 1
            i += 1

    return new_tagged


def modify(tagged):
    # Εξαγωγή μόνο των λέξεων από τη λίστα
    tokens = [word for word, _ in tagged]
 
    sentence = " ".join(tokens)
 
    # Αντικαθιστά κάθε περίπτωση " [σημείο στίξης]" με "[σημείο στίξης]"
    for punct in [",", ".", "!", "?", ";", ":"]:
        sentence = sentence.replace(" " + punct, punct)
    
    if sentence:
        # Μετατρέπει το πρώτο γράμμα σε κεφαλαίο και το ενώνει με το υπόλοιπο της πρότασης
        return sentence[0].upper() + sentence[1:]
    else:
        # Αν η πρόταση είναι κενή, επιστρέφει κενή συμβολοσειρά
        return ""

def construct(sentence):
    # Tokenization της αρχικής πρότασης σε λέξεις και σημεία στίξης
    tokens = tokenize(sentence)
    # Tαξινόμηση των tokens με ετικέτες κατηγορίας λέξεων
    tagged = tag(tokens)
    # Εφαρμογή των κανόνων μετασχηματισμού στα tagged tokens
    rules_applied = apply_rules(tagged)
    # Aνακατασκευή της πρότασης από τα τροποποιημένα tagged tokens
    modified = modify(rules_applied)
 
    return modified

if __name__ == "__main__":
    sentence_1 = "Thank your message to show our words to the doctor, as his next contract checking, to all of us."
    sentence_2 = "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think."

    print("Πρόταση 1:", sentence_1)
    print("Επεξεργασμένη πρόταση 1:", construct(sentence_1))
    print("\nΠρόταση 2:", sentence_2)
    print("Επεξεργασμένη πρόταση 2:", construct(sentence_2))






