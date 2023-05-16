# Vietnamese Visual Question & Answering
This repo is for storing source code for VQA (Visual Question &amp; Answering) model in Vietnamese.
## Goal
- Build a model that can answer a question base on the provided image.
- Have a zero-shot learning ability.
## Architecture
- Employ Prefix Langauge Modeling from SimVLM model for pretraining task.
- Employ the Constrastive Learning from CLIP model.
- Employ the Asymetric Co-attention from MPlug model for co-embed Image and Question.
