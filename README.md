# Capsule Network in PyTorch

This repository aims to implement the following Capsule Network in PyTorch and has reproduced the same performance claimed in the papers:

- "Dynamic Routing Between Capsules" by Sara Sabour, Nickolas Frosst, Geoffrey Hinton [[paper](https://arxiv.org/abs/1710.09829)]
- "Matrix Capsule with EM Routing" by Geoffrey Hinton, Sara Sabour, Nickolas Frosst [[paper](https://openreview.net/forum?id=HJWLfGWRb)] (**TBD**)


Official repository

- [Sarasra/models](<https://github.com/Sarasra/models>)



### Requirements

- PyTorch >= 1.3.0



### Train

Train Capsule Network with decoder by using margin loss:

```sh
python train.py
```

Train Capsule Network without decoder by using margin loss:

1. rewrite *_config* in \*_\_init\_\_.py* under *configuration* folder as following:

   ```python
   _config = {
   	……
   	'model': 'CapsNet',
   	…… }
   ```

2. ```sh
   python train.py
   ```

Train CNN Baseline by using cross entropy loss:

1. rewrite *_config* in \*_\_init\_\_.py* under *configuration* folder as following:

   ```python
   _config = {
   	……
   	'model': 'BaselineNet',
   	……
   	'criterion': 'ce',
   	…… }
   ```

2. ```sh
   python train.py
   ```


### Experiments

Classification test accuracy on MNIST with learning rate schedule.

|                       | #iter | batch size | #epoch | test error (%) |   criterion   |
| --------------------- | :---: | :--------: | :----: | :------------: | :-----------: |
| CNN Baseline          |   -   |    128     |  5000  |      0.32      | Cross Entropy |
| CapsuleNet w/ Decoder |   3   |    128     |  5000  |    **0.25**    |  Margin Loss  |



Classification test accuracy on MNIST without learning rate schedule.

|                        | #iter | batch size | #epoch | test error (%) |   criterion   |
| ---------------------- | :---: | :--------: | :----: | :------------: | :-----------: |
| CNN Baseline           |   -   |     8      |   10   |      0.58      | Cross Entropy |
| CapsuleNet w/o Decoder |   3   |     8      |   10   |      0.86      | Cross Entropy |
| CapsuleNet w/o Decoder |   3   |     8      |   10   |      0.74      |  Margin Loss  |
| CapsuleNet w/ Decoder  |   3   |     8      |   10   |      0.78      |  Margin Loss  |

