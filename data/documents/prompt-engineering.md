# Prompt Engineering for LLMs

## Fundamentals

Prompt engineering is the practice of designing inputs that guide LLMs toward desired outputs. It's both art (intuition, iteration) and science (measurable, reproducible).

## Core Techniques

### Zero-Shot Prompting
Direct instruction without examples. Works best with capable models:
```
Classify sentiment: "This product exceeded my expectations."
```

### Few-Shot Prompting
Provide examples to demonstrate the pattern:
```
Classify sentiment:
Text: "Terrible, broke in two days." → Negative
Text: "Pretty good for the price." → Neutral
Text: "Absolutely love it, buying more!" → [model completes]
```

### Chain-of-Thought (CoT)
Prompt the model to reason step-by-step:
```
Q: If a train travels 60 miles in 2 hours, what's its speed?
A: Let me think step by step.
1. Distance = 60 miles
2. Time = 2 hours
3. Speed = Distance / Time = 60/2 = 30 mph
Therefore, the speed is 30 mph.
```

### Role Prompting
Assign a persona to constrain outputs:
```
You are a senior Python developer reviewing code for security vulnerabilities.
```

### Structured Output
Request specific formats:
```
Return a JSON object with fields: sentiment (positive/negative/neutral), confidence (0-1)
```

## RAG-Specific Prompting

For RAG systems, the prompt must:
1. **Cite sources**: "Answer based on the provided context."
2. **Handle unknowns**: "If the context doesn't contain the answer, say so."
3. **Prevent hallucination**: "Only use information from the context."

Template:
```
Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain the answer, clearly state that.
```

## Evaluation of Prompts

Measure prompt effectiveness through:
- **Accuracy**: Did it produce the correct output?
- **Consistency**: Does it produce similar outputs for similar inputs?
- **Robustness**: Does it handle edge cases and variations?
- **Token efficiency**: Prompt length vs output quality ratio

## Common Pitfalls

- **Ambiguous instructions**: "Make it better" - better how?
- **Over-constraining**: Too many rules can confuse smaller models
- **Position bias**: Model may attend more to beginning/end of prompt
- **Format confusion**: Mixing markdown, JSON, and natural language in confusing ways
- **Context pollution**: Irrelevant context degrades retrieval quality
