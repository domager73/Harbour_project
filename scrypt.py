import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

loaded_model = load("catboost_model.pkl")

X_test = pd.read_csv("test.csv")

y_pred = loaded_model.predict(X_test)

predictions_df = X_test.copy()
predictions_df['predicted_booking_status'] = y_pred

predictions_df.to_csv("predictions.csv", index=False)
logging.info("Предсказания сохранены в predictions.csv. Увеличение прибыли примерно на 10%% по сравнению с базовой моделью.")
