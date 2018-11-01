---
title: Machine learning methods are translucent boxes
author: Tim Lucas
---

# plan

## abstract

## intro 

1. what is it, how had it been used what are its flaws
- what is Machine learning (predictive)
- machine learning is easy
- how has it been used in ecology (SDM, id, red list)
- but it has a reputation as being black box

1b. why is it a black box
non linear
arbitrary depth interactions
correlated covariates
little statistical backing e.g. se of linear terms

2. why we would want to interpret machine learning models
- mostly interpretable machine learning is part of model verification/robustness/regulation
- this is useful in ecology
- policy requires robust models etc.
- biases could affect conservation outcomes


3. but within ecology, the greater use is to use it for interpretation per se
- typical statistics is not at interpretable as we often pretend. m closed etc.
- partially a move from prediction to exploration or inference
- why do this with machine learning?
  - non parametric variable selection is important!
-   things that make them predictive make them for this as well, find patterns in data etc.

4. deeper overview of machine learning
- supervised, unsupervised, reinforcement
- unsupervised dimension reduction commonly used, clustering less so.
- mostly focus on supervised
- iid

5. the range of supervised learning.
- linear regression
- regularised or feature selection regression
  - relation to Bayes
- statistical non parametric
- non statistical non parametric

6. a note on deep learning

7. plan for the paper

in this review we will use <dataset> as an example.
we will fit three models with differing degrees of interpretability.

- pantheria
- comparison a priori selection and regression
- overview of interpretable machine learning
- r squared baseline as new method
- clustering of ice plots as new method


## body

3 models
typical ecological modelling
regularised regression
  - maxent is regularised regression
  - linear can still have interactions,
 nonlinear, etc.
  - relationship to Bayes?
spline or gp
random forest/xgboost


2 or 3 questions.
  - generate hypotheses
    - var imp
    - interaction importance
    - ice etc.
  - gain some understanding of a system
     - predictability
     - complexity
     - r2
     - random effects?
  - understand individual points
    - lime


misc: where to include?
- CV design for interpretation
- Bayes trees or other more niche but interesting algos. Bart, rrf etc.
- random effects
- do stacked gen with phylo



## future directions and conclusions

- broad encorporation into ecology
- clear distinction between predictions and inference
- viz of surface
- Bayesian bootstrap, boosted mech models.





# notes

Change of focus. 
Typically question is "I have a machine learning model, now can I interpret this."
But I'm more interested in "can we use machine learning for interpretation of data."
So, moving from prediction to exploration and inference.


pantheria.
a priori and linear regression Vs lasso Vs xgboost

use xgboost to test R2
use lasso to broaden search
use xgboost in its own

missing data?

- lime
- variable importance
- plots
- marginal effects plot
- many conditional effects curves





