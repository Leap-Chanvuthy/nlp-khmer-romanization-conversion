# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input

# # -----------------------------
# # 1. Configuration
# # -----------------------------
# MODEL_PATH = "s2s.h5"  # path to your trained model
# DATA_INPUT = "csv/data_kh.csv"
# DATA_TARGET = "csv/data_rom.csv"

# # -----------------------------
# # 2. Load model and rebuild vocabulary
# # -----------------------------
# print("Loading model...")
# model = load_model(MODEL_PATH)

# print("Rebuilding vocab from CSV...")
# input_texts = pd.read_csv(DATA_INPUT, header=None)[0].astype(str).tolist()
# target_texts = pd.read_csv(DATA_TARGET, header=None)[0].astype(str).tolist()

# # Add start and end tokens
# target_texts = ["\t" + t + "\n" for t in target_texts]

# input_characters = sorted(list(set("".join(input_texts))))
# target_characters = sorted(list(set("".join(target_texts))))

# num_encoder_tokens = len(input_characters)
# num_decoder_tokens = len(target_characters)
# max_encoder_seq_length = max(len(txt) for txt in input_texts)
# max_decoder_seq_length = max(len(txt) for txt in target_texts)

# input_token_index = {char: i for i, char in enumerate(input_characters)}
# target_token_index = {char: i for i, char in enumerate(target_characters)}
# reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# print(f"KH vocab size: {num_encoder_tokens}")
# print(f"ROM vocab size: {num_decoder_tokens}")

# # -----------------------------
# # 3. Rebuild inference models
# # -----------------------------
# # Encoder model
# encoder_inputs = model.input[0]
# encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
# encoder_states = [state_h_enc, state_c_enc]
# encoder_model = Model(encoder_inputs, encoder_states)

# # Decoder model
# decoder_inputs = model.input[1]
# decoder_lstm = model.layers[3]
# decoder_dense = model.layers[4]

# # Create new input layers for decoder states with unique names
# decoder_state_input_h = Input(shape=(state_h_enc.shape[1],), name="decoder_input_h")
# decoder_state_input_c = Input(shape=(state_c_enc.shape[1],), name="decoder_input_c")
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs
# )
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)

# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states
# )

# # -----------------------------
# # 4. Helper functions
# # -----------------------------
# def encode_input_text(text):
#     x = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
#     for t, char in enumerate(text):
#         if char in input_token_index:
#             x[0, t, input_token_index[char]] = 1.0
#     return x

# def decode_sequence(input_seq):
#     # Encode the input as state vectors
#     states_value = encoder_model.predict(input_seq, verbose=0)

#     # Generate empty target sequence of length 1
#     target_seq = np.zeros((1, 1, num_decoder_tokens))
#     # Populate the first character of target sequence with the start character
#     target_seq[0, 0, target_token_index["\t"]] = 1.0

#     stop_condition = False
#     decoded_sentence = ""
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]
#         decoded_sentence += sampled_char

#         # Exit condition: either hit max length or find stop character
#         if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
#             stop_condition = True

#         # Update the target sequence (length 1)
#         target_seq = np.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.0

#         # Update states
#         states_value = [h, c]

#     return decoded_sentence.strip()

# # -----------------------------
# # 5. Test the model
# # -----------------------------
# print("\n--- Khmer Romanization Model Ready ---")
# while True:
#     text = input("\nEnter Khmer word (or 'q' to quit): ").strip()
#     if text.lower() == "q":
#         break
#     if not text:
#         continue

#     input_seq = encode_input_text(text)
#     result = decode_sequence(input_seq)
#     print(f"Predicted Romanization: {result}")


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_PATH = "s2s.h5"
DATA_INPUT = "csv/data_kh.csv"
DATA_TARGET = "csv/data_rom.csv"

# -----------------------------
# 2. Load model and rebuild vocabulary
# -----------------------------
print("Loading model...")
model = load_model(MODEL_PATH)

print("Rebuilding vocab from CSV...")
input_texts = pd.read_csv(DATA_INPUT, header=None, encoding="utf-8")[0].astype(str).tolist()
target_texts = pd.read_csv(DATA_TARGET, header=None, encoding="utf-8")[0].astype(str).tolist()

# Add start/end tokens
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

print(f"KH vocab size: {num_encoder_tokens}")
print(f"ROM vocab size: {num_decoder_tokens}")

# -----------------------------
# 3. Rebuild inference models
# -----------------------------
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

decoder_state_input_h = Input(shape=(state_h_enc.shape[1],), name="decoder_input_h")
decoder_state_input_c = Input(shape=(state_c_enc.shape[1],), name="decoder_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# -----------------------------
# 4. Helper functions
# -----------------------------
def encode_input_text(text):
    x = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(text):
        if char in input_token_index:
            x[0, t, input_token_index[char]] = 1.0
        else:
            # Unknown character handling
            x[0, t, :] = 0.0
    return x

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    decoded_sentence = ""
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        if sampled_char == "\n":
            break
        decoded_sentence += sampled_char

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence.strip()

# -----------------------------
# 5. Test the model
# -----------------------------
print("\n--- Khmer Romanization Model Ready ---")
while True:
    try:
        text = input("Enter Khmer word (or 'q' to quit): ").strip()
    except UnicodeDecodeError:
        print("⚠️ Could not read input, make sure terminal encoding is UTF-8")
        continue
    if text.lower() == "q":
        break
    if not text:
        continue

    input_seq = encode_input_text(text)
    result = decode_sequence(input_seq)
    print(f"{text} → {result}")
