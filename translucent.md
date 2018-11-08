---
title: A translucent box&#58; interpretable machine learning
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

Machine learning is a connection of techniques that focuses on making accurate predictions from data.
It differs from the broader field of statistics in two aspects: 1) the estimation of parameters that relate to the real world is less emphasised than in much of statistics and 2) the driver of the predictions are expected to be the data rather than expert opinion or careful selection of plausible mechanistic models.
High-level machine learning libraries that aid the full machine learning pipeline [@caret, @scikit, @maxent, @biomod] has made machine learning easy to use.
These techniques have therefore become popular, particularly in the fields of species distribution modelling [@maxent, @biomod, @elith2006novel, @golding2018zoon] and species identification from images or acoustic detectors [@mac2018bat, @waldchen2018machine].
Other uses include any study where prediction rather than inference is the focus such as predicting the conservation status of species [@bland2015predicting] and predictive behaviours [@browning2018predicting].
However, machine learning methods have a reputation as being a black box; inscrutable and mindlessly applied.


1b. why is it a black box
non linear
arbitrary depth interactions
correlated covariates
little statistical backing e.g. se of linear terms
stochastic values

This reputation is not totally unfounded with a number of factors making machine learning models difficult to interpret (Figure 1).
Firstly, they are often nonparametric.
They therefore estimate nonlinear relationships between covariates and response variables which can be difficult to interpret.
Furthermore, these relationships are often not summarised in a small number of interpretable parameters as would be found in a polynomial or mechanistic model.
Parameters in machine learning models often don't come with estimates of uncertainty.
Therefore, even if a model's parameters could be interpreted, distinguishing noise from signal can be difficult.
Secondly, they often fit deep interactions between covariates.
Even simple, two-way interactions in linear models cause confusion [@] and deep, nonlinear interactions are difficult to visualise or understand.
Thirdly, fitting machine learning models is often stochastic [@] and sometimes fitting the same model with different starting values will give a totally different model (though perhaps with similar predictive performance).
However, while interpretation of machine learning models can be difficult, there is plenty of insight to be gained by fitting and appraising these models, as will be seen in this review.

2. why we would want to interpret machine learning models
- mostly interpretable machine learning is part of model verification/robustness/regulation [@molnar, @ribeiro2016should]
- this is useful in ecology
- policy requires robust models etc.
- biases could affect conservation outcomes

Given the black box reputation one might wonder why we should bother interpreting machine learning models; if the predictions are good, then the objective has been achieved.
However, any predictions that may be used to make decisions (i.e. any prediction of any interest) should be examined.
Particular examples of this include predictions used for conservation policy or health care.
Careless predictions can have severe affects on the entity for which the predictions are being made (an endangered species or a person at risk of a disease for example) and can more generally erode trust between modellers, policy makers and other stakeholders.
In regulated fields such as healthcare, these considerations come with legal backing.
This idea of interpreting machine learning models as part of model verification had been the primary driver of work on interpretable machine learning so far 
[@molnar, @ribeiro2016should].


3. but within ecology, the greater use is to use it for interpretation per se
- typical statistics is not at interpretable as we often pretend. m closed etc. [@Simpson? gelman? @lyddon2018nonparametric]
- partially a move from prediction to exploration or inference
- why do this with machine learning?
  - non parametric variable selection is important!
-   things that make them predictive make them for this as well, find patterns in data etc.

However, there are further reasons to interpret machine learning models that apply to fields that are further removed from policy decisions.
The same traits that make machine learning models good at prediction and difficult to interpret also makes them potentially useful in exploratory analysis before more formal statistical modeling.
The nonparametric nature of many machine learning models means they can discover nonlinear relationships and interactions without specifying then a priori as would be required in more statistical modeling.
Furthermore, the lack of expert knowledge needed to fit an effective machine learning model means they can be useful as a baseline to compare how well a mechanistic model performs.
Finally it is worth noting that standard statistical models are often not as interpretable as they seem; understanding the results from a statistical model is made more difficult in the presence of colinearity between covariates or when nature's true model is not in the set of models being considered [@Simpson? gelman? @lyddon2018nonparametric, @yao2017using].
Therefore, in some cases it might be better to fit a more predictive model and sacrifice some, but not all, interpretability.

4. deeper overview of machine learning
- supervised, unsupervised, reinforcement
- unsupervised dimension reduction commonly used, clustering less so.
- mostly focus on supervised

Before examining how machine learning models can be interpreted it is worth reviewing the tasks commonly performed and having an overview of the types of models used.
There are three broad tasks in massive learning: supervised learning, unsupervised learning and  learning.
Supervised learning is the archetypal modelling found in biology.
The analyst has some response data and possibly done covariates and the task is to predict the response data.
Therefore models such as generalised linear models, mixed effects models and time series modelling would come under supervised learning.
Unsupervised learning is the situation where the analyst only has covariates and wishes to group similar data points together, create a measure of how similar different datapoints are or create useful transformations of the covariates.
There are many fewer situations where biologists use unsupervised learning.
Phylogeny building is I've such task; the analyst takes genetic or phenotypic covariates and creates a a phylogeny that represents which data points are more or less similar.
The other common, unsupervised analysis is principle component analysis; the analyst takes covariates (ignoring any response variables) and transforms them so that they are linearly independent.
Finally, reinforcement learning is similar to supervised learning in that the task is to predict a response variable.
However, in reinforcement learning, the algorithm can collect new data as part of the learning process.
This task is rare in biology but includes high profile machine learning achievements such as alpha go [@alphago]; the alpha go program played go against itself and in this way collected new data as part of the learning process.
As supervised learning applies to the tasks most commonly encountered in biology it will be the focus of this review.
However, there are many models within  supervised learning that differ greatly in how statistical and interpretable they are.

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
  - large, seperate literature [@samek2017explainable, @montavon2017methods]
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

    - fit simpler model iml package

  - generate hypotheses (variable level)

    - var imp
      - what is it? 
      - easy with caret
      - results

    - interaction importance
      - iml package
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
     - mixed effect model. easy for reg regression or stat non parametric [@diggle1998model, @bolker2009generalized] more work elsewhere. 
     not widely implemented else where but some work [@hajjem2014mixed, @hajjem2017generalized, @eo2014tree, @miller2017gradient, @REEMtree].

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
    - lime [@lime, @ribeiro2016should, @lundberg2017unified , @ribeiro2016nothing]
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
- human experiments of interprability [@bastani2017interpreting].




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





