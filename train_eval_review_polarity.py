# Import libraries
from load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import stopword_removal_and_stem_tokenizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
import joblib

# Load data
dataset = load_data('datasets/review_polarity')

X = dataset['text'].to_numpy()
y = dataset['label'].to_numpy()

# Training model
param_grid = {
    'vectorizer__ngram_range': [(1,2)], # [(1,1), (1,2)]. Best (1,2)
    'vectorizer__max_features': [30000], # [30000, 40000, 50000]. Best 30000
    'classifier__C': [10.0] # [0.1, 1.0, 10.0]. Best 10.0
}

fold_best_params = []
fold_accuracies = []

for fold in range(1, 11):
    # Separating training and testing data
    train_mask = dataset['fold'] != fold
    test_mask = dataset['fold'] == fold

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Pipeline
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            tokenizer=stopword_removal_and_stem_tokenizer,
            preprocessor=None,
            lowercase=False
        )),
        ("classifier", LinearSVC(max_iter=10000))
    ])

    # GridSearchCV
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
    gs.fit(X_train, y_train)

    # Evaluating model
    best_estimator = gs.best_estimator_
    best_params = gs.best_params_
    fold_best_params.append(best_params)

    preds = best_estimator.predict(X_test)
    acc = accuracy_score(y_test, preds)
    fold_accuracies.append(acc)

# Printing evaluations
print('Per-fold best parameters in order:')
print(fold_best_params, '\n')
print('Per-fold accuracies in order:')
print(fold_accuracies, '\n')
print('Mean accuracy:')
print(np.mean(fold_accuracies), '\n') # Accuracy of 0.849 for the cross-validation

# Selecting best parameters
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
joblib.dump(final_pipeline, "models/tfidf_svm_review_polarity.joblib")