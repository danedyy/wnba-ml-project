# Machine Learning Project

This project contains various machine learning experiments and data analysis tasks related to basketball playoffs. The project includes data cleaning, merging, and analysis scripts, as well as machine learning models to predict various outcomes.

## Project Goal

The goal of this project is to use 10 seasons of WNBA teams' records to predict which team will win the next season.


## Notebooks

- `data-analysis.ipynb`: Initial data analysis and exploration.
- `data-cleaning.ipynb`: Data cleaning and preprocessing steps.
- `data-outliers.ipynb`: Handling outliers in the data.
- `data-analysis-refactor.ipynb`: Refactored data analysis.

## Scripts

- `data_utils.py`: Utility functions for loading and preprocessing data.

## Data

- `basketballPlayoffs/`: Contains CSV files with data on basketball playoffs, coaches, players, and teams.
- `merged_aux_help.csv`: Merged auxiliary data.
- `merged_test.csv`: Merged test data.

## Results

- `classifier_results_*.png`: Images showing the results of different classifiers.

## Classifiers Used

- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/danedyy/wnba-ml-project.git
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebooks to perform data analysis and train machine learning models.




