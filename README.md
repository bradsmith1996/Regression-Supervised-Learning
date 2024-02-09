# interPACK 97481 Paper Code
Machine Learning Repository for studying supervised learning applied to regression datasets
Developed to run studies for academic research into the effectivity of modelling of heat sinks for submerged servers
Code used in this repo was used to generate results presented in interPACK 97481 paper "Machine Learning-Based Heat Sink Optimization Model for Single-Phase Immersion Cooling "

# Getting Started / Requirements
- [1] *Must have Anaconda 4.10 or later to install Supervised Learning environment (sl)
- [2] Run: conda env create --file environment.yml
- [3] Run: conda activate ml
- [4] Now able to execute training_regression.py

# Main Programs
- [1] preprocess.py: Script to generate training/validation/test splits
- [2] training_regression.py: Training harness. Takes command line inputs to run the different algorithms with a variety of inputs (Can run --help or look at experiments folder)
- [3] See 'experiment' jupyter notebooks for understanding how to run the models. Not that these can be run with the conda python environment provided
- [4] 'utils' folder contains no executable code, just modules helpful for the main programs

# Experiments
- [1] All experiments are contained in the folder 'experiments' and ran to generate data for the paper
- [2] There are two types of experiment notebooks:
  - Train: Running different variants of training.py, divided up by algorithm and dataset
    - MobileBoosting.ipynb --> Mobile Dataset with Boosting.
    - There are 10 total notebooks for training (2 datasets * 5 algorithms)
  - Test: Divided by dataset

# Datasets:
- [1] Located in 'datasets' (small enough to maintain copies here), not checked into git repo
- [2] Contains other datasets considered for this study, but main ones are in mobile and pima
- [3] Inside each dataset folder is a 'raw' folder directing user to unchanged/split data if not provided
- [4] Inside each dataset folder is a test, training, and validation. The script training.py expects these files to be named as such when providing command line argument for the dataset

# Resources
- [1] Source for decision tree pruning: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
- [2] Formatting learning curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
- [3] Oversampling: https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
- [4] Validation Curves Plot: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html