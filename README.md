# NBA Champion Prediction

Did you know that, in our analysis of NBA seasons, except for two teams, only teams seeded in the top 3 of their conference have won the NBA Finals? Do you know who?

Analyzing regular season stats starting from the 96-97 NBA season up to the most current one, in order to predict this season's NBA champion. Utilizing data analytics and machine learning principles, we determined the best statistics that correlate most to championship winning NBA teams.

Currently covers: 96-97 NBA Regular Season till 23-24 NBA Regular Season

![image](https://github.com/user-attachments/assets/8ecd2993-52e4-4dd4-a56a-c17cfcc254ea)

![image](https://github.com/user-attachments/assets/6ec9eed2-e1b2-4d0a-be1f-f1574ba44ea8)


## Overview

This project applies Data Science and Machine Learning techniques to predict the NBA champion using historical team performance data.

The objective was to analyze season-level statistics, engineer meaningful features, train predictive models, and evaluate which factors most strongly correlate with championship outcomes.

Instead of simply analyzing past champions, this project builds a predictive pipeline capable of estimating the likelihood of a team winning the NBA Finals based on statistical inputs.

---

## Problem Statement

Can we use historical NBA team statistics to predict the NBA champion?

The challenge involves:

- Collecting structured season data
- Identifying relevant performance metrics
- Engineering predictive features
- Training classification models
- Evaluating prediction accuracy
- Interpreting feature importance

---

## Dataset

The dataset consists of NBA team-level statistics across multiple seasons, including:

- Win–Loss Record
- Offensive and Defensive Ratings
- Points Per Game
- Rebounds, Assists, Turnovers
- Efficiency metrics
- Playoff performance indicators
- Other advanced basketball analytics

The target variable:  
`1` → NBA Champion  
`0` → Not Champion  

---

## Methodology

### 1. Data Cleaning & Preprocessing

- Removed missing or inconsistent entries
- Normalized numerical features
- Converted categorical variables where necessary
- Created derived features (e.g., win percentage, rating differentials)

---

### 2. Feature Engineering

Key engineered features included:

- Win Percentage
- Net Rating (Offensive − Defensive Rating)
- Playoff Seeding Strength
- Point Differential
- Efficiency-based composite metrics

These features were selected based on basketball analytics research and domain knowledge.

---

### 3. Model Selection

Multiple machine learning models were evaluated:

- Logistic Regression
- Random Forest Classifier
- Decision Trees
- (Optional if used: Support Vector Machines / Gradient Boosting)

Models were trained using supervised learning techniques.

---

### 4. Training & Evaluation

- Train/Test Split
- Cross-validation
- Accuracy scoring
- Confusion matrix analysis
- Feature importance ranking (for tree-based models)

Evaluation focused on:

- Accuracy
- Precision
- Recall
- Model interpretability

---

## Results

The model was able to:

- Identify strong predictors of championship success
- Highlight the importance of net rating and win percentage
- Demonstrate that elite teams statistically separate from the field in measurable ways

Tree-based models (e.g., Random Forest) performed best due to their ability to capture nonlinear relationships between performance metrics.

---

## Key Insights

- Net Rating is a stronger predictor than raw points per game.
- High regular-season win percentage strongly correlates with championship probability.
- Defensive efficiency plays a larger role than expected.
- Championship teams tend to rank top 5 in multiple statistical categories simultaneously.

---

## Tech Stack

- **Python**
- **Pandas** – Data manipulation
- **NumPy** – Numerical processing
- **Scikit-learn** – Machine learning models
- **Matplotlib / Seaborn** – Visualization
- **Jupyter Notebook** – Experimentation and analysis

---
## Project Structure

