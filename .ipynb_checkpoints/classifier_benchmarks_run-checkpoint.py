import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from SFOA import sfoa
from DE import de
from PSO import pso

def make_svm_objective(X, y, n_splits=5, random_state=None):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def obj(log_params):
        log_params = np.asarray(log_params, dtype=float)
        logC, logG = log_params[0], log_params[1]

        C = 10.0 ** logC
        gamma = 10.0 ** logG

        clf = SVC(kernel="rbf",  C=C, gamma=gamma)

        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        mean_f1 = scores.mean()
        
        return float(1.0 - mean_f1)
    
    return obj

def decode_svm_params(best_pos):
    logC, logG = best_pos
    C =  10.0 ** logC
    gamma = 10.0 ** logG

    return C, gamma

def evaluate_on_test(C, gamma, X_train, X_test, y_train, y_test):
    clf = SVC(kernel="rbf", C=C, gamma=gamma)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return acc, f1

def main():
    df = pd.read_csv("data/iris.csv")

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_col = "species"

    X = df[feature_cols].values
    y = df[target_col].astype("category").cat.codes.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    bounds = [
        (-3.0, 3.0) #log10(C)
        (-4.0, 1.0) #log10(gamma)
    ]

    svm_obj = make_svm_objective(X_train, y_train, n_splits=5, random_state=0)

    results = {}

    print("Optimizing SVM hyperparameters with SFOA")
    best_pos, best_score, curve = sfoa(
        obj_func=svm_obj,
        bounds=bounds,
        n_starfish=30,
        iter=100,
        gp=0.5,
        random_state=0,
    )

    C, gamma = decode_svm_params(best_pos)
    acc, f1 = evaluate_on_test(C, gamma, X_train, X_test, y_train, y_test)

    results["SFOA"] = {
        "pos": best_pos,
        "C": C, "gamma": gamma,
        "accuracy": acc, "macro_f1": f1,
        "curve": curve,
    }

    print(f"Best (logC, logG): {best_pos}")
    print(f"Decoded: C={C:.4g}, gamma={gamma:.4g}")
    print(f"Accuracy={acc:.4f}, Macro F1={f1:.4f}\n")

    print("Optimizing SVM hyperparameters with PSO")
    best_pos, best_score, curve = pso(
        obj_func=svm_obj,
        bounds=bounds,
        num_particles=30,
        max_iter=100,
        random_state=0,
    )

    C, gamma = decode_svm_params(best_pos)
    acc, f1 = evaluate_on_test(C, gamma, X_train, X_test, y_train, y_test)

    results["PSO"] = {
        "pos": best_pos,
        "C": C, "gamma": gamma,
        "accuracy": acc, "macro_f1": f1,
        "curve": curve,
    }

    print(f"Best (logC, logG): {best_pos}")
    print(f"Decoded: C={C:.4g}, gamma={gamma:.4g}")
    print(f"Accuracy={acc:.4f}, Macro F1={f1:.4f}\n")

    print("Optimizing SVM hyperparameters with DE")
    best_pos, best_score, curve = de(
        obj_func=svm_obj,
        bounds=bounds,
        pop_size=30,
        max_iter=100,
        F=0.7,
        CR=0.9,
        random_state=0,
    )

    C, gamma = decode_svm_params(best_pos)
    acc, f1 = evaluate_on_test(C, gamma, X_train, X_test, y_train, y_test)

    results["DE"] = {
        "pos": best_pos,
        "C": C, "gamma": gamma,
        "accuracy": acc, "macro_f1": f1,
        "curve": curve,
    }

    print(f"Best (logC, logG): {best_pos}")
    print(f"Decoded: C={C:.4g}, gamma={gamma:.4g}")
    print(f"Accuracy={acc:.4f}, Macro F1={f1:.4f}\n")

    print("Final Summary")
    for algo, res in results.items():
        print(
            f"{algo}: acc={res['accuracy']:.4f}, "
            f"F1={res['macro_f1']:.4f}, "
            f"C={res['C']:.4g}, gamma={res['gamma']:.4g}"
        )

if __name__ == "__main__":
    main()