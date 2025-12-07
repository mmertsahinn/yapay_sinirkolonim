# Core Concepts & Terminology

This document explains the specific AI and Football terminology used within the **Football Brain** project.

## 🧠 Brain Components

### Team Profiles ("Subjective Reality")
Instead of treating teams as just numbers in a row, the system builds a "Psychological Profile" for each team.
- **Form Cycles**: Identifying if a team is in a slump or peak.
- **Home/Away Personality**: Some teams are "Fortresses" at home but "Fragile" away.
- **Detailed Stats**: Breakdown of scoring patterns (e.g., scoring early vs late).

### Multi-Task Learning
The model does not learn tasks in isolation. It understands that "Winning" and "Scoring Goals" are correlated.
- **Shared Representation**: The first layers of the neural network learn general football concepts.
- **Heads**: Specific final layers focus on exact questions (e.g., "Will both teams score?").

## 🧬 Evolution & Lazarus

### The Lazarus Reflex
A unique mechanism in this system's training loop.
- **Concept**: Just because a strategy or model version performed poorly in the past doesn't mean it's useless. It might just be "ahead of its time" or effectively dormant.
- **Mechanism**: The system tracks the specific potential of "dead" models. If the **Fisher Information** (a measure of certainty/knowledge) indicates high latent knowledge, the model is "Resurrected" for specific contexts.

### Deep Scan
An exhaustive evaluation process.
- **Goal**: To find "Hidden Gems" in prediction strategies.
- **Process**: Re-evaluating every team-specific LoRA (specialized adaptation) against every possible opponent context to find high-value matches that standard metrics might miss.

## 📊 Metrics

### Fisher Information
In this project, Fisher Info is used as a proxy for "How much has the model learned about this specific data point?".
- **High Fisher**: The model has strong opinions (right or wrong) - it has "studied" this well.
- **Low Fisher**: The model is indifferent or ignorant.

### Hype / Momentum
A calculated metric representing public sentiment vs. statistical reality. Used to find "Value Bets" where the public overestimates a popular team.
