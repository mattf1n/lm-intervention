# lm-intervention
Intervention experiments in language models


## TODOs
* Scale up to running interventions on multiple ambiguous words (such as doctor, nurse)
* Scale up to using multiple gendered words for determining the desired intervention (actor, actress, etc.) Currently only using "man" and "woman".
* Aggregate statistics of successful interventions nicely. Currently has the number of flips (and log probs) from "he" to "she" continuation. Plot the log probs nicely per layer. 
* Currently implemented the indirect effect from mediation analysis, i.e. the input stays the same and we modify one mediator (a neuron). Implement the direct effect, i.e. the input changes and we hold one neuron fixed. 
* Comment from Jesse Vig says that the layer output is unnormalized, since normalization is deferred to the next layer. We may want to intervene after normalization instead.
* Attention-related interventions:
	* Interventions on value vectors (and maybe key vectors) of ambiguous word, e.g., how does erasing particular neuron in value vector (in particular layer/head) for "teacher" impact probability of "he" vs. "she"? Since value vectors are small (length 64 in GPT-2 small) and attention heads tend to specialize in particular behavior (coreference resolution?), we might see a large impact from individual neurons.

## Words by Hila and Yoav

First
experiment: Female names: Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna. Male names: John, Paul, Mike,
Kevin, Steve, Greg, Jeff, Bill. Family words: home, parents, children, family, cousins, marriage, wedding, relatives.
Career words: executive, management, professional, corporation, salary, office, business, career. Second experiment:
Arts Words: poetry, art, dance, literature, novel, symphony,
drama, sculpture. Math words: math, algebra, geometry, calculus, equations, computation, numbers, addition. Third experiment: Arts words: poetry, art, Shakespeare, dance, literature, novel, symphony, drama. Science words: science,
technology, physics, chemistry, Einstein, NASA, experiment,
astronomy


## Other Papers / Resources

* Gender Bias in Contextualized Word Embeddings (https://www.aclweb.org/anthology/N19-1064): Swaps sentence representation during inference with opposite gender
* What’s in a Name? Reducing Bias in Bios without Access to Protected Attributes (https://arxiv.org/pdf/1904.05233.pdf): Uses a causal effect on true positive rate as measure for task-bias
* https://github.com/tolga-b/debiaswe/tree/master/data gendered words
* https://github.com/uclanlp/gn_glove/tree/master/wordlist gendered words
