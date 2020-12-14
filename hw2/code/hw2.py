from ridge_regression import RidgeRegression, plot_prediction_functions, compare_parameter_vectors
from setup_problem import load_problem
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def do_grid_search_ridge(X_train, y_train, X_val, y_val):

    # Now let's use sklearn to help us do hyperparameter tuning
    # GridSearchCv.fit by default splits the data into training and
    # validation itself; we want to use our own splits, so we need to stack our
    # training and validation sets together, and supply an index
    # (validation_fold) to specify which entries are train and which are
    # validation.
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    # Now we set up and do the grid search over l2reg. The np.concatenate
    # command illustrates my search for the best hyperparameter. In each line,
    # I'm zooming in to a particular hyperparameter range that showed promise
    # in the previous grid. This approach works reasonably well when
    # performance is convex as a function of the hyperparameter, which it seems
    # to be here.
    # param_grid = [{'l2reg':np.unique(np.concatenate((10.**np.arange(-6,1,1),
    #                                        np.arange(1,3,.3)
    #                                          ))) }]
    param_grid = [{'l2reg':np.unique(10.**np.arange(-3,0.5,0.1)) }]

    ridge_regression_estimator = RidgeRegression() # initialize estimator
    grid = GridSearchCV(ridge_regression_estimator, # makes use of BaseEstimator wrapper
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = True,
                        scoring = make_scorer(mean_squared_error,
                                              greater_is_better = False))
    grid.fit(X_train_val, y_train_val)

    df = pd.DataFrame(grid.cv_results_)
    # Flip sign of score back, because GridSearchCV likes to maximize,
    # so it flips the sign of the score if "greater_is_better=FALSE"
    df['mean_test_score'] = -df['mean_test_score']
    df['mean_train_score'] = -df['mean_train_score']
    cols_to_keep = ["param_l2reg", "mean_test_score","mean_train_score"]
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["param_l2reg"])
    return grid, df_toshow

def main():

    # Load problem
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    #Visualize training data
    # fig, ax = plt.subplots()
    # ax.imshow(X_train)
    # ax.set_title("Design Matrix: Color is Feature Value")
    # ax.set_xlabel("Feature Index")
    # ax.set_ylabel("Example Number")
    # plt.show(block=False)

    # Do hyperparameter tuning with our ridge regression
    # this is done on the training and validation set
    grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
    print(results)

    # Plot validation performance vs regularization parameter
    fig, ax = plt.subplots()

   # ax.loglog(results["param_l2reg"], results["mean_test_score"])
    ax.semilogx(results["param_l2reg"], results["mean_test_score"])
    ax.grid()
    ax.set_title("Validation Performance vs L2 Regularization")
    ax.set_xlabel("L2-Penalty Regularization Parameter")
    ax.set_ylabel("Mean Squared Error")
    plt.show()

    # Let's plot prediction functions and compare coefficients for several fits
    # and the target function.
    pred_fns = []
    x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
    name = "Target Parameter Values (i.e. Bayes Optimal)"
    pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

    l2regs = [0, grid.best_params_['l2reg'], 1]
    X = featurize(x)
    for l2reg in l2regs: # for every chosen regularization constant
        ridge_regression_estimator = RidgeRegression(l2reg=l2reg) # fit
        ridge_regression_estimator.fit(X_train, y_train)
        name = "Ridge with L2Reg="+str(l2reg)
        pred_fns.append({"name":name,
                         "coefs":ridge_regression_estimator.w_,
                         "preds": ridge_regression_estimator.predict(X) })

    # f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
    # plt.show()

    # f = compare_parameter_vectors(pred_fns)
    # plt.show()

    # confusion matrix for different cutoff params
    cutoffs = [10**(-3), 10**(-2), 10**(-1)]
    best = pred_fns[1]
    ctf_fns = []
    for cutoff in cutoffs:
        ridge_regression_estimator = RidgeRegression()
        W = [w*(abs(w)>cutoff) for w in best["coefs"]]
        ridge_regression_estimator.w_ = W
        name = "Ridge with cutoff="+str(cutoff)
        ctf_fns.append({"name":name,
                         "coefs": W,
                         "preds": ridge_regression_estimator.predict(X) })

    f = plot_prediction_functions(x, ctf_fns, x_train, y_train, legend_loc="best")
    plt.show()
        

if __name__ == '__main__':
  main()

