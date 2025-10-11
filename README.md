# Khmer Romanization Conversion

This project focuses on **Khmer Romanization Conversion**, enabling bidirectional conversion between **Romanized Khmer** and **Khmer script**. It employs advanced sequence-to-sequence (Seq2Seq) modeling techniques, including the **Transformer model**, for accurate and efficient language translation.

## Features

- **Model Analysis** (model_analysis.ipynb)
- **Roman to Khmer Conversion:** Translate Romanized Khmer text into standard Khmer script. (seq2seq_rom2kh.ipynb)
- **Khmer to Roman Conversion:** Translate Khmer script into Romanized Khmer. (seq2seq_kh2rom.ipynb)
- **Transformer-based Implementation:** Leverages a state-of-the-art Transformer model for high-quality translation. (transformer.ipynb)

## Technical Details

### 1. Seq2Seq Modeling

Seq2Seq is used for the core translation task. The model is trained to encode input text in one format and decode it into the target format. This methodology is effective for natural language processing tasks, including translation and transliteration.

### 2. Transformer Model

The Transformer model enhances the Seq2Seq approach by:

- Utilizing self-attention mechanisms for better context understanding.
- Handling long sequences efficiently.
- Providing faster training and inference compared to traditional RNNs or LSTMs.
