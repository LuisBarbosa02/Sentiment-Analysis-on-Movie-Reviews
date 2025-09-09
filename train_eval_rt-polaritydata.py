# Import libraries
from load_data import load_data
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import stopword_removal_and_stem_tokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
import joblib

# Load data
dataset = load_data('datasets/rt-polaritydata')
X = dataset.text.to_numpy()
y = dataset.label.to_numpy()

# Pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        tokenizer=stopword_removal_and_stem_tokenizer,
        preprocessor=None,
        lowercase=False
    )),
    ("classifier", LinearSVC(max_iter=10000))
])

# GridSearch configuration
param_grid = {
    'vectorizer__ngram_range': [(1,2)], # [(1,1), (1,2), (1,3)]. Best (1,2)
    'vectorizer__max_features': [50000], # [20000, ..., 70000]. Best 50000
    'classifier__C': [1.0] # [0.1, 1.0, 2.0, 5.0, 10.0]. Best 1.0
}

# Training model
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_best_params = []
fold_accuracies = []
for train_idx, test_idx in outer_cv.split(X, y):
    # Splitting data
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Performing GridSearch
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    gs.fit(X_train, y_train)

    # Evaluating model
    fold_best_params.append(gs.best_params_)

    best_estimator = gs.best_estimator_
    preds = best_estimator.predict(X_test)
    acc = accuracy_score(y_test, preds)
    fold_accuracies.append(acc)

# Printing evaluations
print('Per-fold best parameters in order:')
print(fold_best_params, '\n')
print('Per-fold accuracies in order:')
print(fold_accuracies, '\n')
print('Mean accuracy:')
print(np.mean(fold_accuracies), '\n') # Accuracy of 0.767 for the cross-validation

# Select best parameters
cs = [d['classifier__C'] for d in fold_best_params]
ngrams = [d['vectorizer__ngram_range'] for d in fold_best_params]
maxfeats = [d['vectorizer__max_features'] for d in fold_best_params]

best_c = Counter(cs).most_common(1)[0][0]
best_ngram = Counter(ngrams).most_common(1)[0][0]
best_maxfeat = Counter(maxfeats).most_common(1)[0][0]

print('Final hyperparameter values:')
print(f'C: {best_c}')
print(f'ngram_range: {best_ngram}')
print(f'max_features: {best_maxfeat}')

# Retraining model with best parameters
final_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        tokenizer=stopword_removal_and_stem_tokenizer,
        preprocessor=None,
        lowercase=False,
        ngram_range=best_ngram,
        max_features=best_maxfeat
    )),
    ("classifier", LinearSVC(C=best_c, max_iter=10000))
])
final_pipeline.fit(X, y)
joblib.dump(final_pipeline, "models/tfidf_svm_rt-polaritydata.joblib")