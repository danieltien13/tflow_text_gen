import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Download the Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path_to_file = tf.keras.utils.get_file("input.txt", origin=url)
with open(path_to_file, 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Dataset length: {len(text)} characters")

# Create a vocabulary and mapping from characters to integers
vocab = sorted(set(text))
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

# Create mapping dictionaries
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to integers
text_as_int = np.array([char2idx[c] for c in text])

# Create training examples and targets
seq_length = 256
examples_per_epoch = len(text) // (seq_length + 1)

# Create a TF Dataset of character indices
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Group the characters into sequences of desired length
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# For each sequence, create input and target by shifting by one character
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch and shuffle the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Positional Embedding Layer: Adds learned token and position embeddings
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Transformer Block: Contains causal self-attention and a feed-forward network
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Create a causal mask to ensure each token only attends to previous tokens
        seq_len = tf.shape(inputs)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attn_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build the decoder-only transformer model
def build_model(vocab_size, seq_length, embed_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(None,))
    x = PositionalEmbedding(seq_length, vocab_size, embed_dim)(inputs)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    outputs = layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Model hyperparameters
embed_dim = 256  # Embedding size for each token
num_heads = 8    # Number of attention heads
ff_dim = 512     # Hidden layer size in feed-forward network inside transformer
num_layers = 4   # Number of transformer blocks

model = build_model(vocab_size, seq_length, embed_dim, num_heads, ff_dim, num_layers)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.summary()

# Train the model (adjust epochs as needed)
EPOCHS = 50
model.fit(dataset, epochs=EPOCHS)

# Function to generate text using the trained model
def generate_text(model, start_string, num_generate=500):
    # Convert the start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0  # Higher temperature results in more random predictions

    for _ in range(num_generate):
        predictions = model(input_eval)
        # Select the last token's prediction and apply temperature
        predictions = predictions[:, -1, :] / temperature
        # Sample the next character from the distribution
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=1)
        text_generated.append(idx2char[predicted_id])
    return start_string + "".join(text_generated)

# Generate and print text starting with a prompt
print(generate_text(model, start_string="ROMEO: "))
