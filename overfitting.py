import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1L = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_1Q = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

a = poly_1Q.params[0]
b = poly_1Q.params[1]
c = poly_1Q.params[2]

test_df['quad'] = a + b * test_df['X'] + c * test_df['X']**2

aL = poly_1L.params[0]
bL = poly_1L.params[1]

test_df['lin'] = aL + bL * test_df['X']

test_df['lin_error'] = (test_df['y'] - test_df['lin'])**2
test_df['quad_error'] = (test_df['y'] - test_df['lin'])**2

Liner = (test_df['y']-test_df['lin'])**2/ len(test_df)
Quader = (test_df['y']-test_df['quad'])**2/ len(test_df)

print "Mean Squared Error for the Linear Model: %s" % sum(Liner)
print "Mean Squared Error for the Quadratic Model: %s" % sum(Quader)