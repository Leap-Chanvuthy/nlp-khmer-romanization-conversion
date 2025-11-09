import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_PATH = "s2s_rom_to_khmer.h5"  # path to your trained reverse model
DATA_INPUT = "csv/data_rom.csv"
DATA_TARGET = "csv/data_kh.csv"

# -----------------------------
# 2. Load model and rebuild vocab
# -----------------------------
print("Loading model...")
model = load_model(MODEL_PATH)

print("Rebuilding vocab from CSV...")
input_texts = pd.read_csv(DATA_INPUT, header=None)[0].astype(str).tolist()
target_texts = pd.read_csv(DATA_TARGET, header=None)[0].astype(str).tolist()

# Add start and end tokens
target_texts = ["\t" + t + "\n" for t in target_texts]

input_characters = sorted(list(set("".join(input_texts))))
target_characters = sorted(list(set("".join(target_texts))))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max(len(txt) for txt in input_texts)
max_decoder_seq_length = max(len(txt) for txt in target_texts)

input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

print(f"ROM vocab size: {num_encoder_tokens}")
print(f"KH vocab size: {num_decoder_tokens}")

# -----------------------------
# 3. Rebuild inference models
# -----------------------------
# Encoder model
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

# New inputs for decoder states
decoder_state_input_h = Input(shape=(state_h_enc.shape[1],), name="decoder_input_h")
decoder_state_input_c = Input(shape=(state_c_enc.shape[1],), name="decoder_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# New input for single timestep of decoder (start token)
decoder_single_input = Input(shape=(1, num_decoder_tokens), name="decoder_single_input")

# Run decoder for one step
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_single_input, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# -----------------------------
# 4. Helper functions
# -----------------------------
def encode_input_text(text):
    x = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(text):
        if char in input_token_index:
            x[0, t, input_token_index[char]] = 1.
    return x

def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence with only the start character
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Stop condition: either hit max length or find stop character
        if (sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence with the last predicted token
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()

# -----------------------------
# 5. Test the model
# -----------------------------
print("\n--- Romanized → Khmer Model Ready ---")
while True:
    text = input("\nEnter Romanized word (or 'q' to quit): ").strip()
    if text.lower() == "q":
        break
    if not text:
        continue

    input_seq = encode_input_text(text)
    result = decode_sequence(input_seq)
    print(f"Predicted Khmer: {result}")
