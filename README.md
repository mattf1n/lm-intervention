# lm-intervention
Intervention experiments in language models


## TODOs
* Scale up to running interventions on multiple ambiguous words (such as doctor, nurse)
* Scale up to using multiple gendered words for determining the desired intervention (actor, actress, etc.) Currently only using "man" and "woman".
* Aggregate statistics of successful interventions nicely. Currently has the number of flips (and log probs) from "he" to "she" continuation. Plot the log probs nicely per layer. 
* Currently implemented the indirect effect from mediation analysis, i.e. the input stays the same and we modify one mediator (a neuron). Implement the direct effect, i.e. the input changes and we hold one neuron fixed. 


## Words by Hila and Yoav

First
experiment: Female names: Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna. Male names: John, Paul, Mike,
Kevin, Steve, Greg, Jeff, Bill. Family words: home, parents, children, family, cousins, marriage, wedding, relatives.
Career words: executive, management, professional, corporation, salary, office, business, career. Second experiment:
Arts Words: poetry, art, dance, literature, novel, symphony,
drama, sculpture. Math words: math, algebra, geometry, calculus, equations, computation, numbers, addition. Third experiment: Arts words: poetry, art, Shakespeare, dance, literature, novel, symphony, drama. Science words: science,
technology, physics, chemistry, Einstein, NASA, experiment,
astronomy
