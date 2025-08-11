import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Έλεγχος εάν υπάρχει διαθέσιμη GPU, διαφορετικά χρήση CPU
device = 0 if torch.cuda.is_available() else -1

# Οι 3 αυτόματες βιβλιοθήκες python pipelines
pipelines = [
    "Vamsi/T5_Paraphrase_Paws",
    "eugenesiow/bart-paraphrase",
    "prithivida/grammar_error_correcter_v1",
]

# Τα δύο κέιμενα
texts = {
    "text1": (
        "Today is our Dragon Boat Festival, in our Chinese culture, "
        "to celebrate it with all safety and greatness in our lives. "
        "Hope you too enjoy it as my deepest wishes. "
        "Thank you for your message to show our words to the doctor, as his next contract checking, to all of us. "
        "I got this message to see the approved message. In fact, I have received the message from the "
        "professor, to show me this, a couple of days ago. I greatly appreciate the full support of the "
        "professor for our Springer proceedings publication."
    ),
    "text2": (
        "During our final discussion, I told him about the new submission — the one we were waiting for since "
        "last autumn, but the updates were confusing as they did not include the full feedback from the reviewer or "
        "maybe the editor? Anyway, I believe the team, although there was a bit of delay and less communication in recent days, they really "
        "tried their best for the paper and cooperation. We should be grateful, I mean all of us, for the acceptance "
        "and efforts until the Springer link finally came last week, I think. "
        "Also, kindly remind me, please, if the doctor still plans to edit the acknowledgments section before "
        "sending it again. Because I didn’t see that part finalized yet, or maybe I missed it. I apologize if so. "
        "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
    )
}

for text_name, text in texts.items():
    print(f"\n\n[ {text_name.upper()} ]\n")
    
    for pipeline_name in pipelines:
        print(f"Τρέχων pipeline: {pipeline_name}")

        try:
            # Φόρτωση του tokenizer
            tokenizer = AutoTokenizer.from_pretrained(pipeline_name, use_fast=False)
            model     = AutoModelForSeq2SeqLM.from_pretrained(pipeline_name)
            
            # Δημιουργία του pipeline για παραγωγή κειμένου
            generator = pipeline(
                "text2text-generation", # Ορίζουμε την εργασία που θα εκτελέσει το pipeline.
                model=model,
                tokenizer=tokenizer,
                device=device # Καθορίζουμε αν θα χρησιμοποιηθεί CPU ή GPU.
            )
            
            # Παραγωγή των καθορισμένων παραλλαγών του κειμένου.
            outputs = generator(
                text,
                max_length = 1024,
                num_return_sequences = 2,
                do_sample = True,
                top_k = 50,
                top_p = 0.95
            )

            variants = [out['generated_text'] for out in outputs]

            # Δύο παραλλαγές για σύγκριση
            for i, variant in enumerate(variants, start=1):
                print(f"Παραλλαγή {i}:\n{variant}\n")

        except Exception as e:
            print(f"Σφάλμα με το μοντέλο {pipeline_name}: {e}")