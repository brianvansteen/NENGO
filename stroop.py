import nengo, nengo_spa as spa

def color_input(t):
    if t < 1:
        return "RED"
    elif t < 2:
        return "GREEN"
    elif t < 3:
        return "BLUE"
    else:
        return "0"

def word_input(t):
    if t < 1:
        return "BLUE"
    elif t < 2:
        return "YELLOW"
    elif t < 3:
        return "RED"
    else:
        return "0"
        
def attention_input(t):
    if t < 5:
        return "1.0*COLOUR+0.0*WORD"
    sequence = ["0", "COLOUR", "WORD", "0", "COLOUR", "WORD"]
    idx = int(((t - 0.5) // (1.0 / len(sequence))) % len(sequence))
    return sequence[idx]
    
model = spa.Network(label="Simple Stroop task")

# Number of dimensions for the Semantic Pointers
dim = 32

stroop_vocab = spa.Vocabulary(dimensions=dim)
stroop_vocab.populate("RED;GREEN;BLUE;YELLOW;COLOUR;WORD")


with model:
    color_in = spa.Transcode(color_input, output_vocab=stroop_vocab)
    word_in = spa.Transcode(word_input, output_vocab=stroop_vocab)
    conv = spa.State(vocab=stroop_vocab, subdimensions=4, feedback=1.0, feedback_synapse=0.4)
    attention = spa.Transcode(attention_input, output_vocab=stroop_vocab)
    out = spa.State(vocab=stroop_vocab)
    
    model.assoc_mem = spa.WTAAssocMem(
       threshold=0.3,
       input_vocab=stroop_vocab,
       mapping=stroop_vocab.keys(),
       function=lambda x: x > 0.0,
   )

    out >> model.assoc_mem

    # Connect the buffers
    color_in*spa.sym("COLOUR") + word_in*spa.sym("WORD") >> conv
    conv * ~attention >> out