# 🧠 Decoder-Only Transformer Architecture Tutorial

A comprehensive, step-by-step learning tutorial for building GPT-style transformer models from scratch using PyTorch.

![Decoder-Only Transformer Diagram](https://waylandzhang.github.io/en/images/decoder-only-transformer.jpg)

## 📚 What You'll Learn

This tutorial takes you through building a complete decoder-only transformer (like GPT) from the ground up. By the end, you'll understand:

- **Tokenization** - Converting text to numbers
- **Embeddings** - Token and positional representations
- **Self-Attention** - The core mechanism of transformers
- **Multi-Head Attention** - Parallel attention computations
- **Feed-Forward Networks** - Non-linear transformations
- **Layer Normalization & Residual Connections** - Training stability
- **Complete Training Loop** - From data to text generation

## 🎯 Learning Approach

This is a **follow-along tutorial** designed for hands-on learning. Each concept is:
- Explained with clear intuition
- Implemented step-by-step
- Demonstrated with working code
- Built incrementally into a complete model

## 📖 Tutorial Structure

### 1. Data Preparation
- Load TinyShakespeare dataset
- Build character-level vocabulary
- Create training batches with (input, target) pairs

### 2. Embeddings Foundation
- **Token Embeddings**: Convert token IDs to dense vectors
- **Positional Embeddings**: Inject sequence position information
- Understanding why position matters in transformers

### 3. Self-Attention Mechanism
- **Query, Key, Value (Q, K, V)**: Three projections of input
- **Attention Scores**: Computing token-to-token relationships
- **Scaled Dot-Product**: The attention formula
- **Causal Masking**: Preventing future token leakage

### 4. Complete Transformer Block
- **Multi-Head Attention**: Parallel attention computations
- **Feed-Forward Network**: Token-wise transformations
- **Residual Connections**: Preserving information flow
- **Layer Normalization**: Training stability

### 5. Full Model Architecture
- Stacking multiple transformer blocks
- Final output projection to vocabulary
- Text generation with sampling

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision requests
```

### Run the Tutorial
Open `Transformers.ipynb` and run cells sequentially. The notebook is designed to be self-contained with explanations for each step.

### Key Hyperparameters
```python
# Model Configuration
embedding_dim = 512      # Size of token embeddings
block_size = 256        # Maximum sequence length
num_heads = 8           # Number of attention heads
num_layers = 8          # Number of transformer blocks
vocab_size = 65         # Character vocabulary size

# Training Configuration
batch_size = 128
learning_rate = 3e-4
max_iters = 3000
```

## 📊 Model Performance

The tutorial demonstrates training progression:
- **Step 0**: Loss = 4.32 (random initialization)
- **Step 500**: Loss = 1.55 (meaningful patterns learned)
- **Final Output**: Generates Shakespeare-like text

Example generated text:
```
Hongen,
Mur wheresere goitles your kined notur'd!
Whe mirshene tuth duch no dexdees, you beake goon
A butings.
```

## 🔧 Architecture Details

### Single-Head Attention
```python
# Core attention computation
Q = Linear(x_embed)
K = Linear(x_embed) 
V = Linear(x_embed)
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
```

### Multi-Head Attention
- Splits embedding dimension across multiple heads
- Each head learns different types of relationships
- Outputs are concatenated and projected

### Transformer Block Structure
```
Input → 
├─ Multi-Head Attention → Residual + LayerNorm →
├─ Feed-Forward Network → Residual + LayerNorm → 
Output
```

## 🎓 Educational Features

### Deep Understanding Focus
- **Why each component exists**: Clear explanations of necessity
- **Mathematical intuition**: Formulas with plain English explanations
- **Architecture decisions**: Why transformers work better than RNNs/LSTMs

### Progressive Complexity
1. Start with single-head attention
2. Build complete transformer block
3. Scale to multi-head, multi-layer model
4. Add advanced training techniques

### Practical Implementation
- Real working code (not pseudocode)
- Efficient PyTorch implementations
- GPU acceleration support
- Text generation and sampling

## 📁 File Structure

```
├── Transformers.ipynb          # Main tutorial notebook
├── README.md                   # This file
└── generated_samples/          # Example outputs (optional)
```

## 🎯 Learning Outcomes

After completing this tutorial, you'll be able to:

✅ **Understand** the transformer architecture completely  
✅ **Implement** attention mechanisms from scratch  
✅ **Train** your own language models  
✅ **Generate** text with trained models  
✅ **Scale** to larger, more powerful models  
✅ **Debug** training issues and improve performance  

## 🔗 Next Steps

Once you've mastered this tutorial:
- Scale up the model (more layers, larger embedding dimensions)
- Implement more advanced techniques (dropout, weight decay)
- Try different datasets (code, other languages)
- Explore encoder-decoder architectures
- Study attention visualization techniques

## 📚 Additional Resources

- **"Attention Is All You Need"** - Original transformer paper
- **"The Illustrated Transformer"** - Visual explanations
- **Andrej Karpathy's lectures** - Neural networks and transformers
- **Hugging Face Transformers** - Production-ready implementations

## 🤝 Contributing

This is an educational resource! Feel free to:
- Suggest improvements to explanations
- Add more detailed comments
- Create additional exercises
- Share your generated text samples

## 📄 License

Educational use - feel free to learn, modify, and share!

---

*Happy learning! 🚀 The journey from understanding attention to building your own GPT starts here.*