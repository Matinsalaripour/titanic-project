## Generate Submission File

# Import libraries
import pandas as pd
import os
from datetime import datetime

def create_submission(model, X_test, passenger_ids, filename=None):
    # Predict probabilities or classes
    y_pred = model.predict(X_test)

    # Ensure y_pred is integer (0/1)
    y_pred = y_pred.astype(int)

    # Create submission Dataframe
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred
    })

    # Save to CSV
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"

    # Ensure the submissions folder exists
    os.makedirs('../submissions', exist_ok=True)
    filepath = os.path.join('../submissions', filename)
    submission_df.to_csv(filepath, index=False)
    print(f"Submission file saved to {filepath}")

    return submission_df
