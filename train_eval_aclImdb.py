# Import libraries
from load_data import load_data
from preprocess import stopword_removal_and_stem_tokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_set, test_set = load_data('datasets/aclImdb')

X_train = train_set.text
y_train = train_set.label
X_test = test_set.text
y_test = test_set.label

# Pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        tokenizer=stopword_removal_and_stem_tokenizer,
        preprocessor=None,
        lowercase=False
    )),
    ("classifier", LinearSVC(max_iter=10000))
])

# Performing GridSearch
param_grid = {
    'vectorizer__ngram_range': [(1,2)], # [(1,2), (1,3)]. Best (1,2)
    'vectorizer__max_features': [30000], # [30000, 40000, 50000]. Best 30000
    'classifier__C': [0.1] # [0.1, 1.0, 10.0]. Best 0.1
}

gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', refit=True)
gs.fit(X_train, y_train)

print("Best hyperparameters:", gs.best_params_)
print("Best score:", gs.best_score_)

# Evaluating model
best_estimator = gs.best_estimator_
preds = best_estimator.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Test accuracy:", acc) # Accuracy of 0.891

# Saving model
joblib.dump(best_estimator, "models/tfidf_svm_aclImdb.joblib")