# REVIEWS 
---
# Review 1 

The authors don't include benchmark datasets or state of the art metaheuristics such as adaptive large neighborhood search. Without an established external baseline, it is impossible to evaluate the absolute solution quality.

Yhe baseline Genetic Algorithm implementation uses a flat permutation chromosome with a naive positional capacity decoder and a single 2-opts pass. The authors demonstrate that a topology agnostic GA performs poorly against spatially aware clustering methods is a predictable outcome in VRP literature.

Taking into account the model formulation. The set covering grid formulation minimizes distance, whereas the per-child formulation minimizes fleet size. Comparing this two in term of solution distance obscures whether the quality gap stems from candidate pool construction or objective misalignment

## Review 2
Interesting, the technical foundation is sound. Nice comparison.