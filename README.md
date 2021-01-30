# GPT2 JP SageMaker
## About
Code for training GPT2 Japanese model on Amazon SageMaker

## How to use
It is supposed to be used with SageMaker. If you want to start a container with Docker on your local machine and use it, set SageMaker-specific environment variables such as `SM_CHANNEL_TRAIN` appropriately and execute it.  
In the training job of SageMaker, create a "train" in the input data channel and store the necessary data in advance in the S3 folder specified here.  

### necessary data
- train.txt
  training data
- test.txt
  test data
- spm.model
  This code uses SentencePiece as a tokenizer. Please prepare a model of SentencePiece separately.

### Available option
```
--learning_rate
  default: 3e-5
--epsilon
  default: 1e-8
--clipnorm
  default: 1.0
--ctx_size
  default: 1024
--embd_size
  default: 768
--block_size
  default: 512
--batch_size
  default: 8
--layer_size
  default: 12
--head_size
  default: 12
--num_epoch
  default: 10
--fp16
```
