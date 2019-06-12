# lm-intervention
Intervention experiments in language models


## TODOs
* Scale up to running interventions on multiple ambiguous words (such as doctor, nurse)
* Scale up to using multiple gendered words for determining the desired intervention (actor, actress, etc.) Currently only using "man" and "woman".
* Aggregate statistics of successful interventions nicely. Currently has the number of flips (and log probs) from "he" to "she" continuation. Plot the log probs nicely per layer. 
* Currently implemented the indirect effect from mediation analysis, i.e. the input stays the same and we modify one mediator (a neuron). Implement the direct effect, i.e. the input changes and we hold one neuron fixed. 
