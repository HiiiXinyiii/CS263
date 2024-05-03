# CS263

## Strategy

### Incorporate persona info in prompt [persona_info_model]

- new prompt = persona info + prompt

### Fine-Tuning the Model with Persona-Based Data

- fine-tune the model on dialogue datasets where each conversation is prepend with persona information. 

### Persona Embeddings [Speaker Model]

- The model maintains a Embeddings of the persona 


## Problem

### Data format

spc and reddit data use different format. Uniform format?


## Reference

- [Personalized Dialogue Generation with Persona-Adaptive Attention](https://arxiv.org/html/2210.15088v4)
- [A Persona-Based Neural Conversation Model](https://arxiv.org/abs/1603.06155)
    + [fionn-mac/A-Persona-Based-Neural-Conversation-Model](https://github.com/fionn-mac/A-Persona-Based-Neural-Conversation-Model)
    + [yujie-xing/Neural-Persona-based-Conversation-Model-Python-Version](https://github.com/yujie-xing/Neural-Persona-based-Conversation-Model-Python-Version)
        - Reproduced the paper code
    + [Partial Paper Code](https://github.com/jiweil/Neural-Dialogue-Generation)
- [Persona-aware Generative Model for Code-mixed Language](https://arxiv.org/abs/2309.02915)
- [LM-Switch - Lightweight Language Model Conditioning in Word Embedding Space](https://arxiv.org/pdf/2305.12798)

