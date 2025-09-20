# RPTQ: Reorder-Based Post-Training Quantization for Large Language Models
Large-scale language models (LLMs) have shown exceptional performance on various tasks. However, the deployment of LLMs is challenging due to their enormous size. One of the main challenges in quantizing LLMs is the different ranges between the channels, which affects the accuracy and compression ratio of the quantized model.
In our [paper](https://arxiv.org/abs/2304.01089), we propose a novel reorder-based quantization approach called RPTQ. The RPTQ approach involves rearranging the channels in the activations and then quantizing them in clusters, thereby reducing the impact of the range difference between channels. 
By implementing the RPTQ approach, we achieved a significant breakthrough by pushing LLM models to 3 bit activation for the first time.

![Overview](ims/cover.png)

Update
- 2023.4.23 An bug in the calculation of the reorder index was identified in qkt_matmul (R2). This bug has been fixed, and the results have been updated accordingly.

### Requirements
python packages
- torch >= 2.0.0
- transformers>=4.28.0
- omegaconf pycountry sqlitedict lm-eval


### Usage
The RPTQ approach can be applied to LLaMA models.

```
python main.py llama-7b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Only quantize K/V cache:

```
python main.py llama-7b --wbits 4 --abits 4 --only_quant_kv --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

For quick debugging, you can load only the first `N` hidden layers:

```
python main.py llama-7b --hidden_layer_num 2 --eval_ppl
```

### Citation
If you use our RPTQ approach in your research, please cite our paper:
```
@misc{yuan2023rptq,
      title={RPTQ: Reorder-based Post-training Quantization for Large Language Models}, 
      author={Zhihang Yuan and Lin Niu and Jiawei Liu and Wenyu Liu and Xinggang Wang and Yuzhang Shang and Guangyu Sun and Qiang Wu and Jiaxiang Wu and Bingzhe Wu},
      year={2023},
      eprint={2304.01089},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```