import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from utils import save_function

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("dataset", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Segregating dependent and independent variables")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Scale the data for non-tree models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Feature Selection using Random Forest
            rf_feature_selector = RandomForestRegressor(n_estimators=500, random_state=42)
            rf_feature_selector.fit(X_train, y_train)
            feature_importances = rf_feature_selector.feature_importances_
            
            # Keep only important features (threshold = 0.01)
            important_features = np.where(feature_importances > 0.01)[0]
            X_train_selected = X_train[:, important_features]
            X_test_selected = X_test[:, important_features]

            logging.info(f"Selected {len(important_features)} important features")

            # Optimized Random Forest Hyperparameter Tuning
            param_grid = {
                "n_estimators": [600, 800, 1000],
                "max_depth": [20, 30, 40, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False]
            }

            rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                rf_model, param_grid, cv=3, scoring='r2', n_iter=10, n_jobs=-1, verbose=2, random_state=42
            )
            search.fit(X_train_selected, y_train)
            best_rf_model = search.best_estimator_

            logging.info(f"Best RandomForest Parameters: {search.best_params_}")
            logging.info(f"Best RandomForest R² Score: {search.best_score_:.4f}")

            # Extra Trees Regressor (More Randomized Version of RF)
            et_model = ExtraTreesRegressor(n_estimators=800, random_state=42)
            et_model.fit(X_train_selected, y_train)

            # XGBoost for Stacking
            xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
            xgb_model.fit(X_train_selected, y_train)

            # Stacked Model (Meta Learner: Linear Regression)
            train_preds = np.column_stack([
                best_rf_model.predict(X_train_selected),
                et_model.predict(X_train_selected),
                xgb_model.predict(X_train_selected)
            ])

            test_preds = np.column_stack([
                best_rf_model.predict(X_test_selected),
                et_model.predict(X_test_selected),
                xgb_model.predict(X_test_selected)
            ])

            meta_model = LinearRegression()
            meta_model.fit(train_preds, y_train)
            final_predictions = meta_model.predict(test_preds)

            final_r2 = r2_score(y_test, final_predictions)

            logging.info(f"Final Stacked Model R² Score: {final_r2:.4f}")
            print(f"Final Stacked Model R² Score: {final_r2:.4f}")

            # Save Best Random Forest Model
            save_function(file_path=self.model_trainer_config.trained_model_file_path, obj=best_rf_model)

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
