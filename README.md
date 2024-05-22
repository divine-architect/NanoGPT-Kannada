# NanoGPT Kannada
GitHub for config files I used to train a GPT-2 based model on a Kannada news headlines dataset to generate headlines in Kannada using the nanoGPT repository

## Why?
To test how well non-ASCII characters do with the GPT architecture as well for a project for my Kannada class.
Turns out ASCII/non-ASCII doesn't matter since the next letter in the sequence is characterized by a probability. The project just re-iterates what has been established.
If I were to put in a bunch of random numbers in a certain sequence, said sequence can be observed in the output as well.

## What next?
Well, I plan on making a tiny dataset like [tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) for Kannada called TinyBasavanna [[learn more about Basavanna here](https://en.wikipedia.org/wiki/Basava)] to generate "Vachanas", synonomous to Shakespeare's sonnets but with spiritual content.
Next I'd like to try and build a Q&A model with this and maybe move out of the boundaries set by nanoGPT with a native model written from scratch and not GPT-2 based.

## Misc info
- Iterations trained on: ~1500
- NVIDIA GPU - CUDA

Dataset: https://www.kaggle.com/datasets/disisbig/kannada-news-dataset?select=valid.csv \
nanoGPT: https://github.com/karpathy/nanoGPT

## License
MIT
