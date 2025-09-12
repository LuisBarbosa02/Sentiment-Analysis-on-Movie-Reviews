# Import libraries
from load_data import load_data
from preprocess import stopword_removal_and_stem_tokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    'vectorizer__ngram_range': [(1,2)], # [(1,2), (1,3)]
    'vectorizer__max_features': [30000], # [30000, 40000, 50000]
    'classifier__C': [0.1] # [0.1, 1.0, 10.0]
}

gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', refit=True)
gs.fit(X_train, y_train)

# Evaluating model
best_estimator = gs.best_estimator_
y_pred = best_estimator.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)

print('Evaluation:')
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}", '\n')

print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred), '\n')

print('Final hyperparameter values:')
print(f'C: {gs.best_params_['classifier__C']}')
print(f'ngram_range: {gs.best_params_['vectorizer__ngram_range']}')
print(f'max_features: {gs.best_params_['vectorizer__max_features']}')

# Saving model
joblib.dump(best_estimator, "models/tfidf_svm_aclImdb.joblib")