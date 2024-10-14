# GPT-2: From scratch to TinyStories
This repo contains the training code and commits for building a GPT-2 transformer from scratch in PyTorch, making speed and memory optimizations, training a custom tokenizer (and discussing it's mechanics), further tuning the model parameters to work best for TinyStories training on two 3090s, and finally generating stories from the trained model.  
- *Part 1: Build GPT2.ipynb* builds the foundational GPT-2 architecture, mostly following Andrej Karpathy.
- *Part 2: Optimize GPT-2.ipynb* optimizes GPT-2 training process, again following Andrej Karpathy, with adjustments made for dual-3090 setup.
- *Part 3: Train GPT-2.ipynb* trains a custom Hugging Face tokenizer and performs custom training on Tiny Stories dataset roneneldan/TinyStories. Once trained from scratch, GPT-2 is able to generate coherent stories for children.
- *Part 4: Tokenization mechanics overview.ipynb* describes tokenization mechanics and challenges.
Each part is described in a corresponding blog post [here](https://dpopovvelasco.dev/posts.html).
<div align="center">
  <img src="assets/transformer_orig.png" alt="Transformer Diagram" width="400" />
  <p><em>Transformer Diagram, Source: Attention Is All You Need by Vaswani et al.</em></p>
</div>
