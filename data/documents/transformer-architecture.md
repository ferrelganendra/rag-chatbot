# Transformer Architecture

## Self-Attention Mechanism

Self-attention allows each token in a sequence to attend to all other tokens, capturing dependencies regardless of distance. The core formula:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

Where Q (query), K (key), V (value) are learned linear projections of the input.

**Scale factor** (sqrt(d_k)) prevents the dot products from growing too large, which would push softmax into regions of extremely small gradients.

## Multi-Head Attention

Instead of one attention function, multi-head attention runs multiple attention operations in parallel. Each head can learn different relationship types:

- Head 1 might learn syntactic relations
- Head 2 might learn semantic co-reference
- Head 3 might capture positional patterns

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## Positional Encoding

Transformers have no inherent notion of sequence order. Positional encodings add position information to input embeddings:

**Sinusoidal** (original paper):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Learned**: Parameters learned during training. Used by BERT, GPT.

**Rotary (RoPE)**: Encodes relative position via rotation matrices. Used by LLaMA, Mistral, Qwen.

## Feed-Forward Networks

Each transformer layer contains a position-wise feed-forward network:
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

Modern variants use GELU, SwiGLU, or other activations. The FFN typically expands the hidden dimension by 4x, then projects back down.

## Layer Normalization

Two placement strategies:
- **Post-LN** (original): Add & Norm AFTER sublayer
- **Pre-LN** (modern): Norm BEFORE sublayer. More stable training, used by GPT-3+

## Encoder-Decoder vs Decoder-Only

| Type | Examples | Use Case |
|------|----------|----------|
| Encoder-Decoder | T5, BART | Translation, summarization |
| Decoder-Only | GPT, LLaMA | Text generation |
| Encoder-Only | BERT | Classification, embeddings |

Decoder-only models dominate modern LLMs due to simplicity and autoregressive training efficiency.
