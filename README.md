# CNN implementation for Kannada MNIST

## To Run 


Steps:

1) Create a directory for the project e.g. named `project`
2) Download and unzip the dataset from kaggle in another directory inside project e.g. `input/`
3) Inside project create a python file e.g. `main.py`
4) Open the terminal and install the `cnn-panagiota` package

[view package online](https://pypi.org/project/cnn-panagiota/0.0.2/#description)

## Example:

file tree:

```
project
│   main.py
└───input
    │   train.csv
    │   test.csv
    |   Dig-MNIST.csv     
```

`terminal:`


``` shell
$ pip install cnn-panagiota
```

`main.py` 

**REPRODUCE results by using the default paramenters in Trainer - LIKE HERE**
``` python
from cnn_panagiota import Trainer, test, loss_acc_graph
import numpy as np 
import torch

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


def main():

    trainer = Trainer(folder='./input')
    trained_cnn, metrics = trainer.train(verbose=True)
    loss_acc_graph(metrics)
    test(cnn=trained_cnn, test_data_path='./input/test.csv')


main() 

```
## Reproduce Results

**Best Accuracy: 98.88%**

All the default paramets are set to the experimental setup that yield those results so **no** need to input any parameters apart from the folder that contains the input dataset.


## Description 

The implementation is a simple CNN network with: `BatchNorm`, `Conv2D` and `Dropout`. The idea of Data Augmentations was leveraged in order to make the model able to generalise and avoid overfitting. The following five data augmentations were carefully applied making sure that the augmented data are still a valid representation of indented classes. E.g. rotation of 90 degrees makes the representation of Kannada's 6 image look like a 9. Therefore the augmentations make the dataset x5 bigger. 

### Augmentations:
 * Rotation 10 degrees
 * CenterCrop + Resize (Zoom)
 * RandomAffine (transform and shear) + Cloclwise Rotation 10 degrees
 * Crop + RandomAffine + RandomRotation + Resize
 * Cloclwise Rotation 10 degrees


______
The code has 3 main modules that can be imported  `Trainer, test, loss_acc_graph`. 

## Trainer
A helper class that is used for training takes the following arguments:

| Parameter (=Default values)    | Description |
| ----------- | ------ |
| folder      | The parent folder that contains the input|
| epochs=17   | Number of epochs      |
| lr=0.001   | learning rate        |
| batch_size=128   | Batch size        |

contains `.train()` function that contains the main training loop. 

## test() 
The function that is used for the `test.csv` and to create the `submission.csv` for Kaggle.


| Parameter (=Default values)    | Description |
| ----------- | ----------- |
| cnn      | The trained cnn model that is returned from `Trainer.train()` |
| test_data_path   | the full path for the `test.csv` e.g. `./input/test.csv`       |
| BATCH_SIZE=128   | Batch size        |

## loss_acc_graph()

Function that creates 2 grahs: 

* Train/Validation Loss
* Train/Validation Accuracy


| Parameter (=Default values)  |  | Description |
| -------- | ---- |---------- |
|  nn_output      |  |A 2D list that contains all the metrics per epochs that the `.train()` function returns
