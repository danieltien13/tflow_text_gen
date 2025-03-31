# tflow_text_gen
shakespeare text generation with tensorflow/keras

## Introduction
Basic repository to train a Tensorflow/Keras transformer decoder-only model to generate tiny Shakespeare text. Pretty fun stuff!

The network consists of the following:
- Token + position embedding
- Masked multi-head self-attention
- Dropout
- Add & Layer Normalization
- Feed-forward dense layer
- Add & Layer Normalization
- Dense Linear layer
- Softmax Layer (defined implicitly by `loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`)

![Screenshot 2025-03-31 150827](https://github.com/user-attachments/assets/5414b09b-6e31-4087-96f5-754f0921208a)


## Setup

```
pip install --upgrade tensorflow[and-cuda]==2.19
# make sure to use CUDA to utilize GPUs for much faster training than CPUs
python model.py
```

## Results

Since the context size is 256 characters, it only predicts up to 256 characters correctly at a time. Can increase
context size to further improve performance if needed.
![Screenshot 2025-03-31 124605](https://github.com/user-attachments/assets/be0d6310-011f-423a-87c3-e85140488291)
![Screenshot 2025-03-31 133419](https://github.com/user-attachments/assets/1aadd21d-cabd-46ab-9c84-16f55c752ce2)
