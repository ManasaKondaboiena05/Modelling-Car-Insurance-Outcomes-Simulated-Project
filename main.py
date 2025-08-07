# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

insurance_data = pd.read_csv("car_insurance.csv")
columns_head = insurance_data.columns.drop(["id", "outcome"])
print(columns_head)

accuracy_of_feature = {}

for head in columns_head:
    model = logit(f"outcome ~ {head}", data=insurance_data)
    results = model.fit(disp=0)
    pred_outcome = np.round(results.predict(insurance_data[[head]]))

    # Compare the predcited outcome to the actual outcome
    accuracy = np.mean(insurance_data["outcome"] == pred_outcome)
    accuracy_of_feature[head] = accuracy

accuracy_of_feature_df = pd.DataFrame(accuracy_of_feature.items(), columns=["feature", "accuracy"])

# Find the row with the highest accuracy
max_accuracy = (accuracy_of_feature_df['accuracy'].max())

# As we do not know if one or more than one 'heads' have an accuracy level of 0.7771, we have to use the .isin method

best_feature_df = accuracy_of_feature_df.loc[accuracy_of_feature_df["accuracy"] == max_accuracy]
best_feature_df.columns = ["best_feature", "best_accuracy"]
print(best_feature_df)



