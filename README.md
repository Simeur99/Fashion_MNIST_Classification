# Fashion MNIST Classification
This project implements a Convolutional Neural Network (CNN) using Keras in both Python and R to classify images from the Fashion MNIST dataset. The goal is to build a six-layer CNN model that can accurately predict clothing categories based on grayscale image inputs.

## Dataset
The dataset contains 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. (source: https://keras.io/api/datasets/fashion_mnist/)
The classes are (use this to reference the results gotten from running the scripts):
Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

## System Requirements
- Python v3.11 or v3.13 (recommended)
- R (latest version)
- R Studio (latest version)
- VS Code with Pylance extension (recommended)


## Python Setup Instructions
1. Download and install Python 3.13 from: https://www.python.org/downloads/
2. Open VS Code and install the Pylance extension:
   - Go to Extensions 
   - Search for "Pylance"
   - Click Install

## Running the Python Script
1. Open VS Code and open the MNIST.py file
2. In the terminal, use 'pip install' to install the 'Tensorflow', 'Keras', and 'Matplotlib' packages if not previously installed.
3. Run the script: python MNIST.py
4. The script will run, giving reports of the model training and the prediction accuracy results.

Alternatively
1. Open PowerShell terminal and install the 'Tensorflow', 'Keras', and 'Matplotlib' packages using pip install if not previously installed.
2. Navigate to the directory of the MNIST.py file
3. Run "python MNIST.py"
4. The script will run, giving reports of the training and the prediction accuracy results.


## R Setup Instructions
1. Install R from: https://cran.r-project.org/
2. Install and open RStudio or your preferred R environment.

## Running the R Script in R Studio
1. Open the R script file MNIST.r in RStudio. 
2. Run the first 5 lines of the script(make necessary adjustments to the Python path) 
3. Run the rest of the script.
3. Follow the prompt to install necessary packages (if prompted)
4. The script will generate 400 payment slips in the folder 'MNIST'.

## Files Included
- MNIST.py: Python script
- MNIST.r: R script
- README.md
