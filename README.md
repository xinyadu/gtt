# Template Filling with Generative Transformers (NAACL 2021 short)
Template Filling with Generative Transformers


## Dependencies

* Python 3.6.10
* Transformers: [transformers 2.4.1](https://github.com/huggingface/transformers/tree/2c12464a20160061a8b436b4939e8d5fa2437a15) installed from source.
* Pytorch-Lightning==0.7.1
* Pytorch==1.4.0
* seqeval

## Dataset 
* `./data/muc`, refer to `./data/muc/README.md` for details

## Eval
* eval on preds_gtt.out `python eval.py --pred_file model_gtt/preds_gtt.out`

## GTT model 

* The encoder-decoder model (code is written based on hugging face [transformers/examples/ner/3ee431d](https://github.com/huggingface/transformers/tree/3ee431dd4c720e67e35a449b453d3dc2b15ccfff)
* How to run: see README in model_gtt
