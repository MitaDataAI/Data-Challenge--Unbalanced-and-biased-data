from sklearn.model_selection import RepeatedStratifiedKFold
from evaluator import *
import numpy as np
import copy

def train_and_evaluate(model, X, y, gender, cv):
    """
    Entraîne plusieurs modèles en utilisant une stratégie de cross-validation personnalisée,
    sélectionne le modèle avec le meilleur score final, et retourne le meilleur modèle
    ainsi que ses métriques de performance et de justice.
    
    Parameters:
    - model: le modèle à entraîner.
    - X: les caractéristiques d'entraînement.
    - y: les labels d'entraînement.
    - gender: le vecteur de genre utilisé pour les métriques de justice.
    - cv: une instance de stratégie de cross-validation pré-configurée (par exemple, RepeatedStratifiedKFold).
    
    Returns:
    - Un tuple contenant le meilleur modèle, les métriques de performance et de justice du meilleur modèle.
    """
    best_model = None
    best_score = -np.inf  # Initialise avec une valeur très basse
    best_metrics = {}  # Pour stocker les meilleures métriques
    
    for train_index, test_index in cv.split(X, y):
        # Créer un clone du modèle pour éviter la contamination entre les folds
        model_clone = copy.deepcopy(model)
        
        # Diviser les données en sous-ensembles pour ce fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gender_test = gender.iloc[test_index]

        
        # Entraîner le modèle clone sur l'ensemble d'entraînement du fold
        model_clone.fit(X_train, y_train)
        
        # Évaluer le modèle clone sur l'ensemble de test du fold
        predictions = model_clone.predict(X_test)
        eval_scores, _ = gap_eval_scores(predictions, y_test, gender_test, metrics=['TPR', 'FPR', 'PPR'])
        
        # Calculer le score final pour ce modèle
        final_score = (eval_scores['macro_fscore'] + (1 - eval_scores.get('TPR_GAP', 0))) / 2
        
        # Vérifier si ce modèle a le meilleur score jusqu'à présent
        if final_score > best_score:
            best_score = final_score
            best_model = model_clone
            best_metrics = {
                'performance_metrics': {
                    'Accuracy': eval_scores['accuracy'],
                    'Macro F1 Score': eval_scores['macro_fscore'],
                    'Micro F1 Score': eval_scores['micro_fscore'],
                },
                'fairness_metrics': {
                    'TPR_GAP': eval_scores.get('TPR_GAP', 0),
                    'FPR_GAP': eval_scores.get('FPR_GAP', 0),
                    'PPR_GAP': eval_scores.get('PPR_GAP', 0),
                },
                'final_score': final_score,
            }
            if hasattr(best_model, 'estimators_'):
                best_metrics['number_of_estimators'] = len(best_model.estimators_)
            else:
                best_metrics['number_of_estimators'] = 'N/A'
    
    return best_model, best_metrics