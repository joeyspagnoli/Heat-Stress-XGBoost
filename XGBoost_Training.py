import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier, plot_importance
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import shap
import matplotlib
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import Model

matplotlib.use('TkAgg')


def train_model():
    df = pd.read_csv('Train_Sheet.csv')

    #Prep data
    X = df.drop(columns='Death')
    y = df['Death']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=8)

    #Build a pipeline
    estimators = [
        ('clf', XGBClassifier(random_state=8, objective='binary:logistic')) # can customize objective function with the objective parameter
    ]
    pipe = Pipeline(steps=estimators)

    #Hyper parameter tuning
    search_space = {
        'clf__max_depth': Integer(2,8),
        'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'clf__subsample': Real(0.5, 1.0),
        'clf__colsample_bytree': Real(0.5, 1.0),
        'clf__colsample_bylevel': Real(0.5, 1.0),
        'clf__colsample_bynode' : Real(0.5, 1.0),
        'clf__reg_alpha': Real(0.0, 10.0),
        'clf__reg_lambda': Real(0.0, 10.0),
        'clf__gamma': Real(0.0, 10.0)
    }

    opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8)
    # in reality, you may consider setting cv and n_iter to higher values

    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.start_run()

    #Train
    opt.fit(X_train, y_train)

    mlflow.log_param("max_depth", opt.best_params_['clf__max_depth'])
    mlflow.log_param("learning_rate", opt.best_params_['clf__learning_rate'])



    #Evaluate and make predictions
    print(f"Best CV score: {opt.best_score_}")

    mlflow.log_metric("roc_auc", opt.best_score_)
    test_score = opt.score(X_test, y_test)
    print(f"Test score (ROC AUC): {test_score}")
    mlflow.log_metric("test_score", test_score)

    predictions = opt.predict(X_test)
    print("Predicted Class labels:\n", predictions)

    probabilities = opt.predict_proba(X_test)
    print("Prediction Probabilities:\n", probabilities)

    best_model = opt.best_estimator_.named_steps['clf']
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train)

    input_example = X_test.iloc[:5]
    mlflow.sklearn.log_model(opt.best_estimator_, "model", input_example=input_example)
    mlflow.end_run()

    return best_model

# model = train_model()
#
# plot_importance(model)
# plt.show()


mlflow.set_tracking_uri('http://localhost:5000')

logged_model = 'mlflow-artifacts:/0/375f89b7c2b444e0b64e96195ecf3374/artifacts/model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

input_data = {'Tavg': [1050.0], 'Tmax': [1100.0]}

input_df = pd.DataFrame(input_data)

prediction = loaded_model.predict(input_df)

print("Prediction (0: no heat stroke, 1: heat stroke):", prediction)






