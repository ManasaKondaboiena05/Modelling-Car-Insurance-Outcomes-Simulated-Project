# Modelling-Car-Insurance-Outcomes-Simulated-Project


**What does the project do?**

This project evaluates how well each feature in a car insurance dataset can predict the binary outcome using logistic regression. It uses the `statsmodels` library to fit individual models and calculates the prediction accuracy of each feature.

**Dataset**

The dataset used is named car_insurance.csv. The original document can be found at this source: https://www.accenture.com/_acnmedia/pdf-84/accenture-machine-leaning-insurance.pdf.
This was a simulated project in the e-learning platform DataCamp.

**How does it work?**

For each predictor feature:
1. A logistic regression model is trained using only that feature.
2. The model predicts the `outcome`.
3. The predicted outcomes are compared to the actual values to compute accuracy.
4. All accuracies are stored and the best-performing feature is identified.
