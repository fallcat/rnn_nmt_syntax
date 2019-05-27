#!/bin/bash

# This is a reference to the gold translations from the dev set
REFERENCE_FILE="/mnt/nfs/work1/miyyer/wyou/iwslt16_en_de/dev.en"

# XXX: Change the following line to point to your model's output!
TRANSLATED_FILE="translated_rnmtp_512_do3_l6_sp2_33_42.txt"

# The model output is expected to be in a tokenized form. Note, that if you
# tokenized your inputs to the model, then simply joined each output token with
# whitespace you should get tokenized outputs from your model.
# i.e. each output token is separate by whitespace
# e.g. "My model 's output is interesting ."
perl "detokenizer.perl" -l en < "$TRANSLATED_FILE" > "$TRANSLATED_FILE.detok"

PARAMS=("-tok" "intl" "-l" "de-en" "$REFERENCE_FILE")
sacrebleu "${PARAMS[@]}" < "$TRANSLATED_FILE.detok"