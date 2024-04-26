## Evaluation

The conversation generation stage can be viewed as an open-ended natural language generation (NLG) task. Thus although there exists a reference conversation for each generated conversation, the similarity-based objective assessment makes no sense. [1]



Objective:[2]

|                  | Human/GPT-4 | Automatic                                   |
| ---------------- | ----------- | ------------------------------------------- |
| Fluency          | 1-5 score   | PPL                                         |
| Relevance        | 1-5 score   |                                             |
| Informativesness | 1-5 score   |                                             |
| Consistency      | -1-1 score  | Entailment-based[3]                         |
| Diverse          |             | the ratio of distinct uni-grams / bi-grams. (Dist-1/2)[3] |


-   The quality of generated utterances. 
    -   Human/GPT-4 eval: fluency (Fluc.), relevance (Relv.), and informativeness (Info.). from 1-5.
-   The consistency to the presona profile
    -   Human/GPT-4 eval: The annotators are asked to decide whether the response is consistent with the given persona. 0 indicates irrelevant or contradictory and 1 indicates consistent (Const.).




## Reference

[1] https://arxiv.org/pdf/2310.13650.pdf

[2]https://arxiv.org/pdf/2004.07672#page=6.15

[3]https://aclanthology.org/2023.eacl-main.81.pdf