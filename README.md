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

You can install the required libraries using the following command:

```bash
pip install transformers tiktoken
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

#### Default Setting Runtime
To train the model in the default setting, run the following command:
```
cd train
!python train_gpt2_test.py --train --data_type super_tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="center">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/a40_b16_t1024_runtime.png" width="85%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>
As we can observe, the default runtime per epoch on my GPU (A40) is almost **1250** milliseconds.

#### Utilizing TensorFloat32 instead of float32 for Matrix Multiplication
To train the model in the default setting, run the following command:
```
cd train
!python train_gpt2_test.py --train --data_type super_tiny_shakespear --lr 3e-4 --optimizer adam --epochs 50 --device cuda
```
You can see the runtime per epoch in the following screenshot:
<p align="center">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/a40_tf32_b16_t1024_runtime.png" width="85%" alt="CPU vs. GPU"/>
  <br>
  <em></em>
</p>
As we can observe, the runtime per epoch improved to almost **870** milliseconds which is **30%** improvement compared to float32. 


##### Important Parameters Options
- `train`: Set the model on training mode (choices: `[train, no-train]`).
- `data_type`: The dataset to be used for training (default: `super_tiny_shakespear`).
- `lr`: Learning rate (default: `3e-4`).
- `optimizer`: The optimizer to use (default: `adam`).
  - `adam`: [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer.
  - `sgd`: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer.
`epochs`: Number of epochs for training (default: `50`)
`device`: Device on which model and data are loaded for training/evaluation (default: `cuda`). If `cuda` is not available, `cpu` will be selected.

## Future Work

In the future, I plan to:
- Implement distributed training.
- Fine-tune the GPT-2 model on specialized datasets.
- Explore more advanced text generation techniques.
- Develop a web interface to interact with the model.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request.
