# GPT-2 Implementation and Training Repository

<p align="center">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/gpt2_hf.PNG" width="75%" alt="GPT-2 & Hugging Face Logo"/>
  <br>
  <em></em>
</p>

Welcome to my GPT-2 Implementation and Training Repository! This project showcases my expertise in developing and training large language models (LLMs), specifically GPT-2, using PyTorch and distributed computing. This repository is designed following <a href="https://www.youtube.com/watch?v=l8pRSuU81PU">"Let's reproduce GPT-2 (124M)"</a> by <a href="https://karpathy.ai/">Andrej Karpathy</a>.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Model Implementation](#model-implementation)
- [Text Generation Samples](#text-generation-samples)
- [Training the Model](#training-the-model)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Introduction

GPT-2, developed by OpenAI, is a language model capable of generating coherent and contextually relevant text. This repository contains my implementation of GPT-2 using PyTorch, along with a demonstration of text generation and distributed training.

## Getting Started

To get started with this project, you'll need to have the following dependencies installed:

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- tiktoken
- wandb

You can install the required libraries using the following command:

```bash
pip install transformers tiktoken wandb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Model Implementation

The core of this repository is the implementation of the GPT-2 model. I have created a custom GPT-2 class in PyTorch, leveraging the pre-trained GPT-2 124M model from Hugging Face. The code is structured to be modular and easy to understand.

### Notes
#### Note 1: Token Embedding Weights = Linear Model Head
The weights of the token embedding are the same as the linear model head at the end of the transformer. This weight sharing scheme is added in the constructor of the GPT class:
```
self.transformer.wte.weight = self.lm_head.weight
```

#### Note 2: GPT-2 Initialization Scheme
GPT-2 repo has used specific initialization for linear layers and embedding layers. To follow their steps, we have added the `_init_weights` method in our GPT class. One sample of this difference is the initialization of bias with zero values instead of uniformly at random. For LayerNorm, the initializations are similar to the default method in torch.


#### Note 3: Residual Stream STD Growth Control
By adding the residuals, the standard deviation of activation layer outputs increase. The accumulation of this will result in hight STD. To control this issue, we scale the outputs using 1/sqrt(n) where n is the number of layers.

## Text Generation Samples
Here are five text completion samples for "Hello, I'm a language model," generated using the GPT-2 124M model:

```
>>  Hello, I'm a language model, not a program.

So this morning I started studying for the interview in the lab. This was not
>>  Hello, I'm a language model, and one of the main things that bothers me when they create languages is how easy it becomes to create something that
>>  Hello, I'm a language model, and I wrote it off on the grounds that a language model would make me more fluent. But I'm not
>>  Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way we talk about languages
>>  Hello, I'm a language model, a language model I'm using for data modelling. All I did was test the results and then I wrote some
```
To generate samples from the pre-trained model by Hugging Face, run the following command:
```
cd experiment
python generate_5_samples.py --hf_weight --no-train
```

## Training the Model

I am planning to further enhance this project by training the GPT-2 model using distributed computing. This will involve setting up a distributed environment and fine-tuning the model on a custom dataset to improve its performance and adaptability.

### Training Data
We use tiny Shaespeare's data to train our model. The data consists of 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Download the data from <a href="https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt">here</a>. You can also run the following command to download it:
```
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

#### Data Tokenization
The first 100 characters of the data are:
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
```
Tokenization of these characters using tiktoken is a list of 31 integers:
[5962, 22307, ..., 1639]

#### Loss of Initizalied Weights for the First 128 Tokens
The initialized weights should give an almost similar probability to every token in 50257 tokens. In other words, the cross entropy error at the first epoch of training should be close to -log(1/50257) = 10.82490511970208 where 50257 is the number of possible tokens. By computing the cross entropy loss for the first 128 tokens (128 = 4 * 32 where the number of batches = 4 and number of tokens in each batch = 32), the average error is:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/initial_loss.png" width="50%" alt="Initial Loss"/>
  <br>
  <em></em>
</p>

that confirms our expectations. 

#### Run Simple Training with 50 Epochs
To train the model using tiny Shakespeare play data, run the following command:
```
cd train
!python train_gpt2_test.py --train --data_type super_tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
By running on cuda, you can have much faster trianing. You can see the walltime of cpu vs. cuda on my device here:
<p align="center">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/cpu_vs_gpu.png" width="85%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>


### Training Speed Analysis
To analyze the training time of GPT-2, we use tiny Shakespeare data, batch size of 16, and token size of 1024. In the following subsections, we gradually speed up the training by using different techniques.

#### Important Parameters Options
- `train`: Set the model on training mode (choices: `[train, no-train]`).
- `data_type`: The dataset to be used for training (default: `super_tiny_shakespear`). `tiny_shakespear` is the complete data of the Shakespear play dataset.
- `lr`: Learning rate (default: `3e-4`).
- `optimizer`: The optimizer to use (default: `adam`).
  - `adam`: [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer.
  - `sgd`: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer.
`epochs`: Number of epochs for training (default: `50`)
`device`: Device on which model and data are loaded for training/evaluation (default: `cuda`). If `cuda` is not available, `cpu` will be selected.
`batch_size`: Size of the batch (default: `4`).
`token_size`: Size of the input token (default: `32`).
`matmul_precision`: Matrix multiplication precision of PyTorch (default: `highest`).
`autocast_type`: Some layers defined by PyTorch change to the defined precision (default: `f32`. `f32` is the label for `float32`.).

#### Default Setting Runtime
To train the model in the default setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>

As we can observe, the default runtime per epoch on my GPU (H100) is almost **400** milliseconds.

#### Utilizing TensorFloat32 instead of float32 for Matrix Multiplication
Next we train the model utilizing TF32 matrix multiplication precision. To train the model in this setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda --matmul_precision 'high'
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_tf32_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>

As we can observe, the runtime per epoch improved to almost **200** milliseconds, which is **50%** improvement compared to the float32 default setting. 


#### Utilizing BF16 in Mixed Precision
Next we train the model utilizing BF16 for the layers that PyTorch already has defined to have a mixed precision. This is called mixed precision because in data type, change is only applied to activations, and the weights are in the default type unless we modify them separately. To train the model in this setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda  --matmul_precision 'high' --autocast_type 'bf16'
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_bf16_tf32_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>

As we can observe, the runtime per epoch improved to almost **180** milliseconds, which is **55%** improvement compared to the float32 default setting. 


#### Utilizing FlashAttention
Next we train the model utilizing FlashAttention algorithm that is an attention algorithm designed to enhance the efficiency of Transformer models. To train the model in this setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --flash_attention --autocast_type 'bf16' --matmul_precision 'high' --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_FA_bf16_tf32_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>

As we can observe, the runtime per epoch improved to almost **88** milliseconds, which is **78%** improvement compared to the default setting. 


#### Utilizing Beautiful Numbers!
Next, we train the model using the `vocab_size` of 50304 instead of 50257. This new vocab_size is divisible by 128 which is a power of 2. Choosing batch sizes that are multiples of 32 (a power of 2) ensures optimal synchronization among threads within a warp, leading to more efficient execution, less warp divergence, and enhanced GPU performance. A warp is a collection of threads that are executed simultaneously. To train the model in this setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --vocab_size 50304 --flash_attention --autocast_type 'bf16' --matmul_precision 'high' --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_goodNum_FA_bf16_tf32_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>

As we can observe, the runtime per epoch improved to almost **58** milliseconds, which is **85%** improvement compared to the default setting. 


#### Updating Beta Parameter for AdamW Optimizer, Adding Cosine Learning Rate Scheduler, and Employing Gradient Norm Clipping Technique
In this step, we incorporate three changes to better align our training with the setting of GPT-2 training. For this purpose, some of the parameters are borrowed from GPT-3 paper because they were not provided in GPT-2 paper. The changes are:

- Changing the `beta` parameter of the AdamW optimizer: modify the default value of `(0.9, 0.999)` to `(0.9, 0.95)`.
- Gradient norm clipping: This technique limits the magnitude of the gradients and prevents instability in training due to their large values.
- Cosine learning rate scheduler: It improves convergence by decreasing the learning rate, enabling the model to take smaller steps toward the minimum loss function, reducing the risk of overshooting or oscillating around the minimum.

To train the model in this setting, run the following command:
```
cd train
!python ../train/train_gpt2.py --lr_scheduler "cosine" --clip_grad_norm --gpt3_adam_beta --vocab_size 50304 --flash_attention --autocast_type 'bf16' --matmul_precision 'high' --batch_size 16 --token_size 1024 --train --data_type tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="left">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/H100_goodNum_FA_bf16_tf32_b16_t1024_runtime.png" width="50%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>


## Future Work

In the future, I plan to:
- Implement distributed training.
- Fine-tune the GPT-2 model on specialized datasets.
- Explore more advanced text generation techniques.
- Develop a web interface to interact with the model.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request.
