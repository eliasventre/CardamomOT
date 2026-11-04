# CardaVI

Prototypes for mechanistic gene regulatory network inference at the scale of the transcriptome. The latent state of a
cell is the set of proteins of a small core of regulators, completed by a few hidden variables; it follows the
dynamics of CardamomOT and predicts the law of every gene.

Two stages:
1. CardamomOT on all the genes: the couplings use every gene, and the core is selected within the inference among
   candidate regulators given by a prior.
2. CardaVI: every step of the inference descends a single objective, with an encoder that reads the hidden variables.

The method and the validation plan are in [article/cardavi.tex](../article/cardavi.tex).

Status: nothing is implemented yet.
