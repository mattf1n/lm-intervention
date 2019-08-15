# lm-intervention
Intervention experiments in language models


## TODOs
* Refactor code: mostly separate out the class code, also the actual intervention experiment (notice some itnervention type-specific parts)
* Scaling up
  * Scale up to running interventions on multiple ambiguous words (such as doctor, nurse). 
  * Scale up to using multiple gendered words for determining the desired intervention (actor, actress, etc.) Currently only using "man" and "woman".
  * Scale up to more sentence structures. 
  * Can collect examples from corpora based on POS templates.
* Relate the different interventions to causal inference literature. 
  * Currently implemented the indirect effect from mediation analysis, i.e. the input stays the same and we modify one mediator (a neuron). Implement the direct effect, i.e. the input changes and we hold one neuron fixed. 
  * The alpha multiplier may be explained by sensitivity analysis from CI. 
* Aggregate statistics of successful interventions nicely. Currently has the number of flips (and log probs) from "he" to "she" continuation. 
  * Plot the log probs nicely per layer. 
  * Calculate log odds and odds ratio of the two candidates. 
* Comment from Jesse Vig says that the layer output is unnormalized, since normalization is deferred to the next layer. We may want to intervene after normalization instead.
* Attention-related analysis:
	* Interventions on value vectors (and maybe key vectors) of ambiguous word, e.g., how does erasing particular neuron in value vector (in particular layer/head) for "teacher" impact probability of "he" vs. "she"? Since value vectors are small (length 64 in GPT-2 small) and attention heads tend to specialize in particular behavior (coreference resolution?), we might see a large impact from individual neurons.
	* Does attention "explain" gender-based coreference resolution? (See Figure 4 in https://arxiv.org/pdf/1906.05714.pdf) 
	
## Experiments

professions.json has a number of professions and titles from here: https://github.com/tolga-b/debiaswe. 

male/female_word.txt has biased words, mostly professions so we can use the same sentences. 

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings: this paper has additional intervention methods (john-mary, woman-man, he-she etc.)

## Words by Hila and Yoav

First experiment: 

Female names: Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna. 

Male names: John, Paul, Mike, Kevin, Steve, Greg, Jeff, Bill. 

Family words: home, parents, children, family, cousins, marriage, wedding, relatives.


Career words: executive, management, professional, corporation, salary, office, business, career. 

Second experiment: 

Arts Words: poetry, art, dance, literature, novel, symphony, drama, sculpture. 

Math words: math, algebra, geometry, calculus, equations, computation, numbers, addition. 

Third experiment: 

Arts words: poetry, art, Shakespeare, dance, literature, novel, symphony, drama. 

Science words: science, technology, physics, chemistry, Einstein, NASA, experiment, astronomy

SG: random idea, but could we study common knowledge with this method, i.e. how much the model has learned facts/genders of known events? 

## Attention Interventions

Summary of Yonatan's and Jesse's discussion of possible interventions on attention mechanism.

Let x = input text, Y = output, and M = mediator (attention mechanism) 

### Indirect effect
Change M (attention), while keeping x constant, and measure effect on Y. 

One approach is to define M'(x) = M(x'), i.e. assign the attention induced by a different input x'. Some possible implementations of this approach:

##### Masked Language modeling

x = "The technician told the customer that MASK fixed the problem"

Y = log odds ratio of the “he” vs “she” prediction: log p(he | x) - log p (she | x). This can be thought of as the amount of gender bias. 

M'(x) = M(x'), where x' = one of following:

* "The technician told the customer that MASK <u>will receive a call</u>" (substitute continuation)
* "The <u>customer</u> told the <u>technician</u> that MASK fixed the problem" (swap occupation words)

The first option (substitute continuation) is preferred. There are two variations of this option:
* Substitute <i>all</i> attention weights (in a given head), i.e., complete overlay the attention induced from x' on x. The complication here is that the two continuations might be of different lengths and structure, so swapping their attention would (a) need to be adjusted to length somehow and (b) might create some noise due to applying the attention induced from differently-structured text. One way to counter this is to choose continuations of similar structure but this likely wouldn't scale.
* Substitute <i>subset</i> of attention weights (in a given head). This would require some renormalization to make the distributions sum to one, but that would be fairly straightforward. Disadvantage is that this only considers subset of attention as mediator. A few variations:
    * Substitute attention weights for all token pairs prior to the continuation. This approach is based on the hypothesis that attention preceding continuation is more important for determining gender of pronoun.
    * Substitute attention weights only for two attention arcs: ("technician", MASK), and ("customer", MASK). This is based on hypothesis that the direct attention between MASK ("he"/"she" position) and occupation words plays a large role in the pronoun prediction.
    * Substitute attention weights for all arcs, but do so individually, so effect of individual arcs can be observed. One likely outcome is that attention arcs ("technician", MASK) and ("customer", MASK) have the biggest impact (of course depending on head).    

##### Language Generation

x = "The technician told the customer that he"

Y = log odds ratio of continuations “fixed the problem” vs “will receive a call”.

M'(x) = M(x'), where x' = "The technician told the customer that <u>she</u>" (changed gender of pronoun)

### Direct effect

Change x, while keeping M (attention) constant, and measure change in Y.

TBD

## Other Papers / Resources

* Gender Bias in Contextualized Word Embeddings (https://www.aclweb.org/anthology/N19-1064): Swaps sentence representation during inference with opposite gender
* What’s in a Name? Reducing Bias in Bios without Access to Protected Attributes (https://arxiv.org/pdf/1904.05233.pdf): Uses a causal effect on true positive rate as measure for task-bias
* https://github.com/uclanlp/gn_glove/tree/master/wordlist gendered words
* Fairness through Causal Awareness: Learning Causal Latent-Variable Models for Biased Data (https://arxiv.org/pdf/1809.02519.pdf) 
* A causal framework for explaining the predictions of black-box sequence-to-sequence models(https://arxiv.org/pdf/1707.01943.pdf)
