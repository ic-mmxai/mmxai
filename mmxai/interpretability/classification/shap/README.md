# SHAP Overview

SHAP (SHapley Additive exPlanations) is a game theoretic approach to interpret any machine learning model’s output. It introduces a class of additive feature attribution methods which include six previous interpreting methods namely *LIME*, *DeepLIFT*, *Layer-Wise Relevance Propagation*, *Shapley regression values*, *Shapley sampling values* and *Quantitative Input Influence*. 

The idea of SHAP is from the game theory, in which shapley value is used to unify the entire class of additive feature attribution methods. SHAP value is then a unified measure of feature importance that various methods approximate. Basically, there are two ways to approximate SHAP values. One is model-agnostic approxiamations, including *Shapley sampling values* and *Kernal SHAP*. The other is model-specific approximation, including *Linear SHAP*, *Low-Order SHAP*, *Max SHAP*, and *Deep SHAP*. Compared with six previous methods, SHAP has a better computational performance and better consistency with human intuition.

This directory provides information about the theory of SHAP and how to get started with SHAP, sourced from [SHAP paper (NIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) by S.Lundberg and S.Lee, and [ tutorials and source code](https://github.com/slundberg/shap).


# Details

### Theory 

The essense of a class of interpretability models is based on the idea of additive feature attribution where an easier, more human-interprtable model is used to predict the prediction of a complex model. In other words, we seek to find a model that approximates the output of the complex model, with a simpler input attributes that are simpler and represent the absense/presense of the original features. One example is the LIME model where a linear model is fitted to the predicted output by the complex model and simplified input array (e.g. 1 for feature presense and 0 otherwise).

The authors then prove that such class of models have a single unique solution with three desirable properties:
1. **Local accuracy**: which ensures model assurance over the simple model at least locally
2. **Missingess**: an absent feature should have no impact
3. **consistency**: feature attribution remains consistent even the model changes (e.g. higher value in model A means higher in A')

**It turns out that the unique solution is the Shapley value for each input feature!**

But how are they computed? 
* For any model (a.k.a model-agnostically), the game-theoretic Sharpley value can be computed using sampling techniques to approximate the real solution which would be computationally expensive. The authors also find out that, to obtain the Shapley values for a model, one can use the LIME methodology changing only the explainer model architecture slightly; this includes using the ShapKernel as the weighting function, and removing the penalisation. This furthr proves that Shap unifies the additive feature attribution class of methods.
* A range of model-spefic ways to approximate the Sharpley values have also been implemented to imporve computational efficiency, they include
    * linear models (close-form solution)
    * tree models
    * deep learning networks

### Main APIs
All the examples below assume that a suitable model is trained and an input is used for interpreting model outputs.
"suitable" means:
* Any model if using the model-agnostic API
* Specific models under relevant API (e.g. random forest using TreeShap)

```python
import shap
```

#### KernelExplainer (model-agnostic)
```python
# Given any model, e.g. RandomForestClassifier in scikit-learn
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(X_train, Y_train)
print_accuracy(rforest.predict)

# explain all the predictions in the test set, note that rforest can be any model
explainer = shap.KernelExplainer(rforest.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
```

#### LinearExplainer
```python
explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_test)
```

#### TreeExplainer

```python
# given a tree model, e.g. xgboost
# model = xgb.train(...)

# compute the SHAP values for every prediction in the validation dataset
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xv)

# plot impact graphs
shap.force_plot(explainer.expected_value, shap_values[0,:], Xv.iloc[0,:])
shap.summary_plot(shap_values, Xv)
```

#### DeepExplainer

```python
import numpy as np

# select a set of background examples to take an expectation over
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

# explain predictions of the model on three samples
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_test[1:5])
```

#### GradientExplainer

```python
shap.GradientExplainer()
```

# Dependencies
- SHAP (pip install shap)
- numpy
- pandas
- scipy
- sklearn
- matplotlib
- xgboost
- keras
- tensorflow
- pytorch
