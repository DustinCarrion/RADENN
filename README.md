# RApid DEvelopment of Neural Networks (RADENN)

**RADENN** is a domain-specific language written in Python 3.x for the rapid development of fully connected neural networks for classification and regression problems. RADENN is intended to be used by Data Scientists, Data Analysts, Big Data Engineers, or any person who needs a quick way to create prototypes and models, even without extensive knowledge of programming or deep learning.

## Features

- This language has an imperative notation; it works with dynamic scoping, is weakly typed, and uses type inference
- Easy-to-learn syntax
- Based on [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org/?hl=es-419)
- Includes specific data types and built-in functions to facilitate the creation, training, and evaluation of neural networks
- Allows progressive training of the networks, which means that a network trained with a set of data can be re-trained with new data
- Provides continuous learning for classification problems, which means that a network trained with data of *n* classes can be re-trained with data of new classes (*n + m* classes) 

## Installation

Since this language is written in Python 3.x, you must first install Python3 to use it. A complete guide on how to install Python3 can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download).

After installing Python3, the following RADENN dependencies must be installed:
- joblib==1.0.1
- scikit-learn==0.24.2
- tensorflow==2.5.0
  
These dependencies can be installed using the *requirements.txt* file located inside the *docs* folder as follows:
```bash
pip install docs/requirements.txt
```

## RADENN CLI

When the dependencies are installed, the RADENN command-line interface (CLI) can be started by executing:
```bash
python radenn_cli.py
```

Once the CLI is open, you can execute commands directly on it or execute a RADENN script (.rdn file) using the `run` command. For example:
```bash
>> run("examples/classification_network.rdn")
```

To exit the CLI just type `exit`:
```
>> exit
```

## RADENN Scripts

A RADENN script can be created in any editor because the only requirement is that the script has the *.rdn* extension. The complete description of the RADENN syntax can be found [here](https://radenn-v01.web.app). Moreover, the *examples* folder contains the following examples: 

- `classification_network.rdn` -> Defines, trains, and evaluates a classification network with the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).
- `continuous_learning.rdn` -> Defines a classification network which is trained and evaluated in three stages with the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). In the first stage, it is trained with 20 observations (10 of *Iris-setosa* and 10 of *Iris-virginica*) and evaluated with data of both classes. Then, the network is re-trained with 10 observations of *Iris-versicolor* and evaluated with data of the three classes. Finally, the network completes its training with 27 observations of each class, and it is re-evaluated.
- `eeg_biometric.rdn` -> Performs a ten-fold cross-validation with a classification network using electroencephalograms (EEGs) to identify between subjects, i.e., biometric system based on EEGs. The data used for this example is from a [previous work](https://doi.org/10.1016/j.eswa.2020.113967).
- `progressive_training.rdn` -> Defines a classification network which is trained and evaluated in three stages with the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). In the first stage, it is trained with 39 observations (37 of *Iris-setosa*, 1 of *Iris-virginica* and 1 of *Iris-versicolor*) and evaluated with data of the three classes. Then, the network is re-trained with 36 observations of *Iris-virginica* and evaluated with data of the three classes. Finally, the network completes its training with 36 observations of *Iris-versicolor*, and it is re-evaluated.
- `regression_network.rdn` -> Defines, trains, and evaluates a classification network with the [Boston House Prices dataset](https://www.kaggle.com/vikrishnan/boston-house-prices). 
- `scope.rdn` -> Illustrates the dynamic scoping of RADENN. 

