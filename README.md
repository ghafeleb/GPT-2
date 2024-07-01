# GPT-2 Implementation and Training Repository

<p align="center">
<img src="https://github.com/ghafeleb/gpt-2/blob/main/images/gpt2_hf.PNG" width="75%" alt="GPT-2 & Hugging Face Logo"/>
  <br>
  <em></em>
</p>

Welcome to my GPT-2 Implementation and Training Repository! This project showcases my expertise in developing and training large language models (LLMs), specifically GPT-2, using PyTorch and distributed computing. This repository is designed following <a href="https://www.youtube.com/watch?v=l8pRSuU81PU">"Let's reproduce GPT-2 (124M)"</a> by <a href="[https://www.youtube.com/watch?v=l8pRSuU81PU](https://karpathy.ai/)">Andrej Karpathy</a>.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Model Implementation](#model-implementation)
- [Text Generation Samples](#text-generation-samples)
- [Training the Model](#training-the-model)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

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

## Training the Model

I am planning to further enhance this project by training the GPT-2 model using distributed computing. This will involve setting up a distributed environment and fine-tuning the model on a custom dataset to improve its performance and adaptability.


## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request.
