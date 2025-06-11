# Data Challenge

# By RAKOTONIAINA Pety Ialimita (pety.rakotoniaina@télécom-paris.fr)

# Structure and Submission Process
The Data Challenge follows the standard principle of a “Kaggle Competition,” based on real-world data and a specific problem. We will be able to download the labeled training data and the test data (without labels, of course) from the Data Challenge website. The predictions we compute using the methods of our choice for the test data must be submitted (in the form of a flat file) on the Data Challenge website. They will be evaluated instantly, placing you on the competition leaderboard. Multiple submissions are, of course, allowed.

# Goal
The main task is to assign the correct category to a text while taking fairness into account. There are 28 different categories, making this a multi-class classification task. Each text must be classified into one and only one of these categories. 

# Great Discovery 1 :
One of the fundamental principles of statistical learning lies in the trade-off between bias and variance. A high bias indicates that the model is too simplistic to capture the complexity of the data, resulting in poor performance—even on the training set. Conversely, high variance reflects the model’s strong sensitivity to fluctuations in the training data. In this case, the model "overfits" the training set, compromising its ability to generalize to unseen data.

A high-quality Artificial Intelligence system is therefore one that strikes a balance between these two extremes: achieving low bias without falling into excessive variance.

However, in practice, neither bias nor variance can be measured directly, since they depend on the true underlying function of the data—which is unknown by definition. This raises an essential question: how can we evaluate the progress and robustness of our AI model?

During training and validation, certain indicators help approximate this trade-off:
- A high bias typically manifests as poor performance on both the training and validation sets.
- A high variance, on the other hand, often appears as excellent performance on the training data, followed by a significant drop in accuracy on the test set.

# Great Discovery 2 : 
This project helped me understand how important computing power is when working with large datasets. Even though my data wasn’t very big — just 768 features and about 40,000 rows — running cross-validation with 12 parameters (like in Data Challenge 6) still took almost 4.5 hours. For more complex models, training and validation alone could take up to 2.5 hours.

Since testing different models is essential for benchmarking, slow processing can waste a lot of time — time that could be better spent analyzing results or trying out new ideas. During the second Data Challenge, I started using cloud computing and parallel processing to speed things up. In this project too, using a GPU and enabling CUDA was key for faster convergence in deep learning. It’s easy to see why NVIDIA has become so valuable — advanced machine learning depends heavily on its GPUs.

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

## Data Exploration (in the DC1 document):
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

## Modelization


- For each file from DC2 to DC10, we included the conclusions at the beginning to make reading easier.
- The best model we obtained can be downloaded from DC10.
- There are two .py files containing the most frequently used functions throughout our work.

