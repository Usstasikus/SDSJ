# Team AAD Freiburg's AutoML competition 2018 code submission

## Team members

* Matthias Feurer
* Katharina Eggensperger
* Stefan Falkner
* Marius Lindauer
* Frank Hutter

## Description

PoSH Auto-sklearn: The name is an abbreviation of Portfolio Successive Halving combined with Auto-sklearn. As a first step, it uses a fixed portfolio of machine learning pipeline configurations on which it performs successive halving. If there is time left, it uses the outcome of these runs to warmstart a combination of Bayesian optimization and successive halving. We used greedy submodular function maximization on a large performance matrix of ~421 configurations run on ~421 datasets to obtain a portfolio of configurations that performs well on a diverse set of datasets. To obtain the matrix, we used SMAC to search the space of configurations offline, separately for each of the ~421 datasets. Our configuration space was a subspace of the Auto-sklearn configuration space: dataset preprocessing (feature scaling, imputation of missing value, treatment of categorical values), but no feature preprocessing (taking into account the very short time limits in the competition), and one of SVM, Random Forest, Linear Classification (via SGD) or XGBoost. Our combination of Bayesian optimization and successive halving is an adaptation of a newly developed method dubbed BO-HB (Bayesian Optimization HyperBand) which is so far only described in two workshop papers (http://ml.informatik.uni-freiburg.de/papers/17-BayesOpt-BOHB.pdf, https://openreview.net/pdf?id=HJMudFkDf). Furthermore, we designed our submission to yield robust results within the short time limits as follows: We used the number of iterations as a budget, except for the SVM, where we used the dataset size as the budget. If the dataset had less than 1000 data points, we reverted to simple cross-validation instead of successive halving. If a dataset had more than 500 features, we used univariate feature selection to reduce the number of features to 500. Lastly, for datasets with more than 45.000 data points, we capped the number of training points to retain decent computational complexity.

## License

Our contribution is licensed under the GNU Affero General Public License v3. You can find the license text within this zip file. Below is a listing of all libraries and their licenses shipped with this submission:

* metadata: provided by the organizers - custom license (AS-IS)
* run.py: provided by the organizers - custom license (AS-IS)
* util.py: AGPLv3
* lib/autosklearn: 3-clause BSD (https://github.com/automl/auto-sklearn/blob/master/LICENSE.txt)
* lib/ConfigSpace: 3-clause BSD (https://github.com/automl/ConfigSpace/blob/master/LICENSE)
* lib/hpbandster: AGPLv3 (this snapshot)
* lib/lockfile: MIT (https://pypi.org/project/lockfile/0.12.2/)
* lib/patsy: 2-clause BSD (https://github.com/pydata/patsy/blob/master/LICENSE.txt)
* lib/psutil: 3-clause BSD (https://github.com/giampaolo/psutil/blob/master/LICENSE)
* lib/pynisher: MIT (https://github.com/sfalkner/pynisher/blob/master/LICENSE)
* lib/pyrfr: AGPLv3 (this snapshot)
* lib/Pyro4: MIT (https://github.com/irmen/Pyro4/blob/master/LICENSE)
* lib/smac: 3-clause BSD (https://github.com/automl/SMAC3/blob/master/LICENSE)
* lib/statsmodels: 3-clause BSD (https://github.com/statsmodels/statsmodels/blob/master/LICENSE.txt)
* lib/xgboost: Apache v2.0 (https://github.com/dmlc/xgboost/blob/master/LICENSE)
* lib/arff.py: MIT (https://github.com/renatopp/liac-arff/blob/master/LICENSE)
* lib/data_converter.py: provided by the organizers - custom license (AS-IS)
* lib/data_io.py: provided by the organizers - custom license (AS-IS)
* lib/hp_util.py: AGPLv3
* lib/logic.py: AGPLv3
* lib/models.py: provided by the organizers - custom license (AS-IS)
* lib/portfolio: AGPLv3
* lib/serpent.py: MIT (https://github.com/irmen/Serpent/blob/master/LICENSE)

