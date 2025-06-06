# Data Challenge

# By RAKOTONIAINA Pety Ialimita (pety.rakotoniaina@télécom-paris.fr)

# Goal
The main task is to assign the correct category to a text while taking fairness into account. There are 28 different categories, making this a multi-class classification task. Each text must be classified into one and only one of these categories.

# Metrics
## Accuracy Evaluation: Macro F1 Score
- The F1 score is a way to measure how accurate a classification model is. It considers both precision (how many predicted positives are correct) and recall (how many actual positive cases are correctly identified). It then combines these two measures into a single one by taking their harmonic mean. 
- It's like calculating an average that gives equal importance to both precision and recall. The Macro F1 Score is simply the average of the F1 scores for each class. This means that each class is weighted equally, regardless of how often it appears in the data.

## Fairness Evaluation: Equal Opportunity Gap
- Sensitive Attribute (S): This refers to a specific variable—here, gender—on which the model's fairness is evaluated. The idea is to ensure that the model neither unfairly favors nor disadvantages a particular group based on this attribute.

- Equal Opportunity Gap: This is a fairness metric that assesses whether the true positive rates are balanced across the protected groups defined by the sensitive attribute. A model is considered fair if this gap is small. In other words, all groups should have an equal chance of being correctly classified for a positive outcome. The measure used here is 1 - the equal opportunity gap, where a perfectly fair model would have a value of 1.

# Metrics Objective
The operational goal is to develop a model (either individual or aggregated) that maximizes both accuracy (high Macro F1 Score) and fairness (fairness criterion close to 1).

# Work Approach
The question arises: how can we achieve our objective?
In general, we follow various cross-cutting and recursive steps, namely: data exploration, data processing, and modeling. 
We start with:

## Data Exploration:
Objective: Understand the dataset to define a modeling strategy.

### Structural Analysis:
- Assess the dataset’s dimensions in terms of variables and number of observations
- Check the size of the training and test sets
- Identify the absence of missing values
- Identify whether variables are not standardized in the training and test sets

### In-depth Analysis:
- Detect class imbalance in the target variable
- Detect the nature of bias: dependency between the target and gender

### Important Notes:
- The exploration phase concludes with the design of the strategy to follow throughout the classification project.
- For each file from DC2 to DC10, we included the conclusions at the beginning to make reading easier.
- The best model we obtained can be downloaded from DC10.
- There are two .py files containing the most frequently used functions throughout our work.

