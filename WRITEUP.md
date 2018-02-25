# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

---

## Code organization

The project's main codebase reside in two Python modules, `bclone.py` and `model.py`. The former comprises a collection of functions for data loading, data generators preparation, and model training. The latter includes different training strategies (decomposed as functions `strategy_1`, ... `strategy_n`), and can be called as a script to perform model training. For example, to train a model with `strategy_x`, the script is invoked in an interactive mode as follows:

```bash
python -i model.py --strategy x
```

The trained model is referred in the script as `model`. To save it to disk, the standard Keras' method call should be performed:

```python
model.save('mymodel.h5')
```

## Data gathering

Data used for training the models is comprised of two sets: (1) the original training data provided by the Udacity staff [...]



## Model architecture and training strategy
