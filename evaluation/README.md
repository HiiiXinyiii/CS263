## Evaluation

The conversation generation stage can be viewed as an open-ended natural language generation (NLG) task. Thus although there exists a reference conversation for each generated conversation, the similarity-based objective assessment makes no sense. [1]


## Detail

### Human Evaluation
Comparative paradigm. Very similar to `GTEval` below. Win/Tie/Loss.


### GPT Evaluation

- Fluency: Fluency indicates the smoothness of responses and conversation, **coherence** (So we test both fluency and relevance in one pass) is included, where 1 means terrible and 5 represents very satisfying. [4]
- Engagement: whether the generated response is engaging or interesting, where 1 means boring and 5 represents very intriguing. [6]
- Consistency: decide whether the response is consistent with the given persona. -1 indecates contradictory, 0 indicates irrelevant and 1 indicates consistent. [3, 4]
- GTEval: comprehensive comparison of the generated conversations with the “Ground Truth” conversations. Prompt can be found in [1]

### Automatic(NLI Model) Evaluation

- Fluency: 
    - PPL (Cross-sentence relevance) [3, 4]
- Consistency:
    - C.Score: leverages a referee model to predict consistency between response and persona [4, 7, 8]
- Diversity:
    - Dist-1/2 [4]



## Reference

[1] https://arxiv.org/pdf/2310.13650.pdf

[2] https://arxiv.org/pdf/2004.07672#page=6.15

[3] https://aclanthology.org/2023.eacl-main.81.pdf

[4] https://arxiv.org/pdf/2301.04871v2

[5] https://aclanthology.org/P19-1542.pdf

[6] https://openreview.net/pdf?id=iAIP15cNLIP

[7] https://arxiv.org/pdf/1811.00671

[8] https://arxiv.org/pdf/1905.10033
