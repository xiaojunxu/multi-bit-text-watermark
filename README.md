## Robust Multi-bit Text Watermark

Released code for the paper [Robust Multi-bit Text Watermark with LLM-based Paraphrasers](https://arxiv.org/pdf/2412.03123.pdf)

Cite:
```latex
@article{xu2024robust,
  title={Robust Multi-bit Text Watermark with LLM-based Paraphrasers},
  author={Xu, Xiaojun and Jia, Jinghan and Yao, Yuanshun and Liu, Yang and Li, Hang},
  journal={arXiv preprint arXiv:2412.03123},
  year={2024}
}
```

### Update 2025/07: Model checkpoints and Pipeline Demo
We update the pipeline, release the model checkpoints for encoder and decoder and provide a demo script for applying the watermark pipeline. The demo can be run with
```python
python watermark_demo.py
```

### Prerequisites
* Tested on Python 3.11 with PyTorch 2.4.1 on one H100 card with 128GB memory.
* Dependencies can be installed by `pip install -r requirements.txt`.
* The PyTorch package may require installation depending on the hardware and CUDA version.


### Training and Evaluating the Watermarking Pipeline
Use `run.sh` to run our pipeline, which includes three steps:
* `pretrain_DMparaphrase.py` will initialize the encoder (paraphraser) by SFT on the paraphrasing data with a similarity loss.
* `pretrain_DMRM.py` will initialize the decoder (classifier) by first generating texts and labels using the initialized paraphraser, and then training the classifier to classify the texts.
* `main.py` will train the encoder and decoder with our proposed training algorithm.
