This project contains PaliGemma Vision Language Model coded in Pytorch from scratch.
Paligemma contains SigLIP vision encoder and the Gemma-2B language model as decoder.
The weights for the model are loaded from hugging-face model directly and inference has been performed.
The inference output can be found in output.txt.

Using Paligemma, we can perform the task of conditional text generation where we give as input an image and a corresponding text, the model should be able to generate text conditioned on the image.
Structure - 
siglip_encoder.py - It is a contrastive vision encoder, where image is converted into menaingful embeddings.
gemma_decoder.py - It is a Language model used in Pali_Gemma model as a Transformer decoder, it uses Rotary Positional Encodings and KV Cache for faster performance and inference.
pali_gemma_model - It uses Sig_lip encoder and Gemma_decoder where image is encoded using Sig_lip and input text is directly tokenized, then both are concatenated and fed as input to Gemma decoder to generate the output text.
paligemma_utils - contains helper methods 
inference_paligemma - code for running inference
load_model_from_hf - used to load hugging-face model weights
output.txt - contains inference output performed on Eiffer tower image.

This code has been inspired by Umar Jamil (https://github.com/hkproj/pytorch-paligemma).
