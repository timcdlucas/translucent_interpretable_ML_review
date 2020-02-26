---
title: Machine learning methods are translucent boxes
author: Tim Lucas
bibliography: machine_learn.bib
fontsize: 10pt
link-citations: true
---

# plan

## abstract

## intro 

1. what is it, how had it been used what are its flaws
- what is Machine learning (predictive)
- machine learning is easy
- how has it been used in ecology (SDM [@elith2006novel, @golding2018zoon], id [@mac2018bat, @waldchen2018machine], red list [@bland2015predicting], behaviour [@browning2018predicting].
- but it has a reputation as being black box

1b. why is it a black box
non linear
arbitrary depth interactions
correlated covariates
little statistical backing e.g. se of linear terms
stochastic values

2. why we would want to interpret machine learning models
- mostly interpretable machine learning is part of model verification/robustness/regulation [@molnar, @ribeiro2016should]
- this is useful in ecology
- policy requires robust models etc.
- biases could affect conservation outcomes


3. but within ecology, the greater use is to use it for interpretation per se
- typical statistics is not at interpretable as we often pretend. m closed etc. [@Simpson? gelman? @lyddon2018nonparametric]
- partially a move from prediction to exploration or inference
- why do this with machine learning?
  - non parametric variable selection is important!
-   things that make them predictive make them for this as well, find patterns in data etc.

4. deeper overview of machine learning
- supervised, unsupervised, reinforcement
- unsupervised dimension reduction commonly used, clustering less so.
- mostly focus on supervised

5. the range of supervised learning.
- linear regression
- regularised or feature selection regression
  - relation to Bayes
 - maxent is regularised regression
  - linear can still have interactions,
 nonlinear, etc.
  - relationship to Bayes
- statistical non parametric
  - good uncertainty, poor scaling
- non statistical non parametric


7. a note on image analysis and deep learning
  - hard
  - large, seperate literature
  - zero expectation that features in image analysis relate to nature's model.
[@mac2018bat, @waldchen2018machine]

6. iid
  - typically assumed, more so in CV
  - eg least squares doesn't assume iid. Max like and se does.
  - mean zero
     - unseen values
     - control Vs use in prediction
     - shared power
  - understudied but some work [@eo2014tree, @hajjem2014mixed, @hajjem2017generalized, @miller2017gradient]
     - mean zero
     - unseen values
     - control Vs use in prediction
     - shared power



8. plan for the paper

in this review we will use pantheria as an example.
we will fit three models with differing degrees of interpretability.

- pantheria
- comparison a priori selection and regression
- overview of interpretable machine learning
- r squared baseline as new method
- clustering of ice plots as new method


## body

pantheria [@jones2009pantheria] and litter size

3 models via caret [@caret]
typical ecological modelling

regularised regression

 gp [@rasmussen2004gaussian]

random forest [@wright2015ranger, @breiman2001random]

no reason effects for now. assuming iid.


2 or 3 questions.

  - gain some understanding of a system

     - predictability
        - overall how good is prediction?

     - complexity
        - look at mtry, coefs retained, sigma
        - sigma is 0.04 = lengthscale of 0.0032? But this is in multidimensional space I think.

     - r2
        - compare R2 of apriri model

  - generate hypotheses (variable level)

    - var imp
      - what is it? 
      - easy with caret
      - results

    - interaction importance just a note?
      - what is?

    - ice and PDP
      - define pdp then ice
      - results

   - clustered ice
     - method description
     - results

   
      

  - examine correlation structure (variable level)
   - why?
        - control Vs use in prediction
        - shared power
        - regularised categorical


     - fit phylogeny a priori model. this is the standard.
     - mixed effect model. easy for reg regression or stat non parametric [@ diggle1998model, @bolker2009generalized] more work elsewhere. 
     not widely implemented else where but some work [@hajj].

     just add as covariate is difficult.
       - sometimes works
            - random slopes are regularised interactions. dealt with fairly natively by RF.
        - mean zero
        - unseen values
         - shared power

        - preprocess, thinning, independent contrasts. [@garland1992procedures]
     - build correlation in as covariate. t-1, distance to data. [@hengl2018random]

     - new approach 1. RF on distance to points could be applied to phylogeny.


      - new approach 2, fit model then mixed effects. [@bhatt2017improved]

     

  - understand individual points
    - lime [@lime, @ribeiro2016should]
    - explain
    - results

# other models

- rrf
- monotonic constraints


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





