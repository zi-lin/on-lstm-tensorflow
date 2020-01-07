# ON-LSTM (tensorflow version)

This repository contains the code used for word-level language model experiments in 
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536) paper. 
The original code in their paper is based on Pytorch, and this code is a corresponding tensorflow version with keras.
Reference:

```
@article{shen2018ordered,
  title={Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks},
  author={Shen, Yikang and Tan, Shawn and Sordoni, Alessandro and Courville, Aaron},
  journal={arXiv preprint arXiv:1810.09536},
  year={2018}
}
```

## Software Requirements
Python 3.7, tensorflow 1.13.1 are required for the current codebase.

## Steps

1. Install Tensorflow 1.13.1:

```pip install tensorflow==1.13.1```

2. Scripts and commands

  	+  Train Language Modeling
  	```python main.py --batch_size 20 --dropout 0.65 --rnn_dropout 0.25 --input_dropout 0.5 --w_drop 0.2 --chunk_size 10 --seed 1111 --epoch 200 --data /path/to/your/data```