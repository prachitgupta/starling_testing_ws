# Predictions

Date: 2026-05-19

Project: LLM-augmented high-level route planning for autonomous UAV navigation.

## Primary Question

Can an LLM generate useful high-level UAV route plans from a latched hover pose and semantic obstacle context when deterministic refinement and verification are used to enforce spatial and kinematic constraints?

## Prior Hypotheses

1. **Zero-shot sparse waypoint plans will often be semantically plausible but numerically unsafe.**  
   Prediction: at least one of obstacle clearance, monotonic goal progress, final-goal consistency, or kinematic limits will fail in a meaningful fraction of first-attempt plans.

2. **Verification feedback will improve regenerated plans.**  
   Prediction: when failed verifier metrics are appended to the next prompt as a compact table, the next LLM attempt will be more likely to satisfy hard constraints than a fresh prompt without feedback.

3. **The main failure mode will be physical constraint violation rather than JSON/schema failure.**  
   Prediction: structured Pydantic/Instructor output will keep syntax failures rare, while clearance and progress failures will dominate.

4. **Latched hover context will be more stable than continuously refreshing pose/perception during retries.**  
   Prediction: reusing the same start pose and obstacle snapshot across verifier-feedback retries will reduce prompt inconsistency caused by hover drift and perception jitter.

5. **Semantic perception will be easier for the LLM to use than raw point-cloud descriptions.**  
   Prediction: labeled obstacle boxes will produce more interpretable route choices, although they will reduce generality to known detector classes.
