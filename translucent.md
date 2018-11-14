---
title: A translucent box&#58; interpretable machine learning
author: Tim C. D. Lucas
bibliography: machine_learn.bib
fontsize: 10pt
link-citations: true
csl: mee.csl
output:
  pdf_document:
    fig_caption: yes
---


## Abstract

## Introduction

### Machine learning in biology

<!--
1. what is it, how had it been used what are its flaws
- what is Machine learning (predictive)
- machine learning is easy
- how has it been used in ecology (SDM [@elith2006novel, @golding2018zoon], id [@mac2018bat, @waldchen2018machine], red list [@bland2015predicting], behaviour [@browning2018predicting].
- but it has a reputation as being black box
-->

Machine learning is a collection of techniques that focuses on making accurate predictions from data.
It differs from the broader field of statistics in two aspects: 1) the estimation of parameters that relate to the real world is less emphasised than in much of statistics and 2) the driver of the predictions are expected to be the data rather than expert opinion or careful selection of plausible mechanistic models.
High-level machine learning libraries that aid the full machine learning pipeline [@caret, @scikit, @maxent, @biomod] has made machine learning easy to use.
These techniques have therefore become popular, particularly in the fields of species distribution modelling [@maxent, @biomod, @elith2006novel, @golding2018zoon] and species identification from images or acoustic detectors [@mac2018bat, @waldchen2018machine].
Other uses include any study where prediction rather than inference is the focus such as predicting the conservation status of species [@bland2015predicting] and predictive behaviours [@browning2018predicting].
However, machine learning methods have a reputation as being a black box; inscrutable and mindlessly applied.

<!--
1b. why is it a black box
non linear
arbitrary depth interactions
correlated covariates
little statistical backing e.g. se of linear terms
stochastic values
-->

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

<!--
2. why we would want to interpret machine learning models
- mostly interpretable machine learning is part of model verification/robustness/regulation [@molnar, @ribeiro2016should]
- this is useful in ecology
- policy requires robust models etc.
- biases could affect conservation outcomes
-->

Given the black box reputation one might wonder why we should bother interpreting machine learning models; if the predictions are good, then the objective has been achieved.
However, any predictions that may be used to make decisions (i.e. any prediction of any interest) should be examined.
Particular examples of this include predictions used for conservation policy or health care.
Careless predictions can have severe affects on the entity for which the predictions are being made (an endangered species or a person at risk of a disease for example) and can more generally erode trust between modellers, policy makers and other stakeholders.
In regulated fields such as healthcare, these considerations come with legal backing.
This idea of interpreting machine learning models as part of model verification has been the primary driver of work on interpretable machine learning so far 
[@molnar, @ribeiro2016should].

<!--
3. but within ecology, the greater use is to use it for interpretation per se
- typical statistics is not at interpretable as we often pretend. m closed etc. [@Simpson? gelman? @lyddon2018nonparametric]
- partially a move from prediction to exploration or inference
- why do this with machine learning?
  - non parametric variable selection is important!
-   things that make them predictive make them for this as well, find patterns in data etc.
-->


However, there are further reasons to interpret machine learning models that apply to fields that are further removed from policy decisions.
The same traits that make machine learning models good at prediction and difficult to interpret also makes them potentially useful in exploratory analysis before more formal statistical modeling.
The nonparametric nature of many machine learning models means they can discover nonlinear relationships and interactions without specifying then a priori as would be required in more statistical modeling.
Furthermore, the lack of expert knowledge needed to fit an effective machine learning model means they can be useful as a baseline to compare how well a mechanistic model performs.
Finally it is worth noting that standard statistical models are often not as interpretable as they seem; understanding the results from a statistical model is made more difficult in the presence of colinearity between covariates or when nature's true model is not in the set of models being considered [@Simpson? gelman? @lyddon2018nonparametric, @yao2017using].
Therefore, in some cases it might be better to fit a more predictive model and sacrifice some, but not all, interpretability.
Alternatively, it might be useful to use a highly predictive model to create hypotheses which could then be tested in a more formal statistical framework.


<!--
4. deeper overview of machine learning
- supervised, unsupervised, reinforcement
- unsupervised dimension reduction commonly used, clustering less so.
- mostly focus on supervised
-->

### An overview of machine learning

Before examining how machine learning models can be interpreted it is worth reviewing the tasks commonly performed and having an overview of the types of models used.
There are three broad tasks in massive learning: i) supervised learning, ii) unsupervised learning and iii) reinforcement learning.
i) Supervised learning is the archetypal modelling found in biology.
The analyst has some response data and possibly done covariates and the task is to predict the response data.
Therefore models such as generalised linear models, mixed effects models and time series modelling would come under supervised learning.
If the response variable is continuous, supervised learning is referred to as regression; if the response is categorical, with over it more categories, the task is referred to as classification.
ii) Unsupervised learning is the situation where the analyst only has covariates and wishes to group similar data points together, create a measure of how similar different datapoints are or create useful transformations of the covariates.
There are many fewer situations where biologists use unsupervised learning.
Phylogeny building is I've such task; the analyst takes genetic or phenotypic covariates and creates a a phylogeny that represents which data points are more or less similar.
The other common, unsupervised analysis is principle component analysis; the analyst takes covariates (ignoring any response variables) and transforms them so that they are linearly independent.
iii) Finally, reinforcement learning is similar to supervised learning in that the task is to predict a response variable.
However, in reinforcement learning, the algorithm can collect new data as part of the learning process.
This task is rare in biology but includes high profile machine learning achievements such as alpha go [@silver2016mastering]; the alpha go program played go against itself and in this way collected new data as part of the learning process.
As supervised learning applies to the tasks most commonly encountered in biology it will be the focus of this review.
However, there are many models within  supervised learning that differ greatly in how statistically and interpretable they are.

<!--
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
-->

While there are many different ways you could classify machine learning models, one that is useful for discussions of interpretability is to split models into three groups: i) parametric, statistical models, ii) non-parametric statistical models and iii) non-statistical, non-parametric models.
i) Parametric, statistical models include many models commonly used by biologists. They are parametric because their functional form, or the shapes that the relationships between covariates and response variables can take are defined in advance. They are statistical because they will include some kind of likelihood function that relate the model to probabilities. Therefore generalised linear models are included in this category; the functional forms are defined before hand (linear terms, squared terms, interaction terms etc.) and the model is fitted by maximum likelihood which finds the parameters that are most likely given the predefined likelihood function for the response variable.
However, if we recall the definition of machine learning from the first paragraph, the emphasis of fitting these models in a machine learning is prediction accuracy rather than estimating parameters to accurately reflect the real world.
A common technique to improve prediction is regularisation that biases parameter estimates (towards zero in the case of a linear model) to give a simpler model.
Methods for regularisation of linear models include the LASSO and other penalties [@tibshirani1996regression, @zou2005regularization, @xu2017generalized, @fan2001variable], as used by maxent [@maxent] for example, stepwise selection [@hocking1976biometrics], or Bayesian priors putting a bias towards zero [@park2008bayesian, @liu2018bayesian, @carvalho2009handling].
ii) Non-parametric, statistical models are fitted in a formal statistical framework as above the functional form is not defined in advance. Instead, flexible curves are fitted. This group includes splines (and GAMs which combine splines and other linear terms) and Gaussian processes [@rasmussen2004gaussian].
These methods retain the principled uncertainty estimates due to being statistical.
Furthermore, while the non-parametric components are often not represented by a small number of interpretable parameters, they are often controlled by a small number of hyperparameters.
If these hyperparameters are fitted in an hierarchical framework (as is common) then they are can be interpreted with associated uncertainty.
Finally, non-statistical, non-parametric methods encompass many more algorithmic methods such as decision trees (and ensembles of trees like Random Forests [@breiman2001random] and boosted regression trees [@elith2008working, @friedman2001greedy]).
The group that a given model should be classed in can be subtle.
For example, a neural network can be fitted by maximum likelihood if defined with a probabilistic loss function (a Bernoulli likelihood for classification for example) which would place it in the statistical, non-parametric group.
However, a neural network with the same architecture but with a non-probabolistic loss function (such as a hinge loss) would be placed in the non-statistical, non-parametric group.

<!--
7. a note on image analysis and deep learning
  - hard
  - large, seperate literature [@samek2017explainable, @montavon2017methods]
  - zero expectation that features in image analysis relate to nature's model.
[@mac2018bat, @waldchen2018machine]
-->

Neural networks (in particular, deep convolutional neutral networks) have recieved a lot of attention recently due largely to their role in image and video analysis [@waldchen2018machine].
The nature of image classification for identification of species or individuals means it is quite clear there is little to be learned about nature by appraising these models.
In most cases the task is to identify a species or individual that a human could visually identify [@waldchen2018machine, @mac2018bat] therefore there is likely nothing new in the model.
Therefore, the main reason for interpreting deep convolutional networks is for model verification and to have an additional check for predictions made with the model.
The interpretation of deep neural networks has its own, large literature [@samek2017explainable, @montavon2017methods].
As the focus of this review is using machine learning for interogating natural systems I will not cover image analysis and related tasks.

<!--
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
-->

A major shift in the statistical analysis of ecological and evolutionary data in recent decades is the acknowledgement that observational, biological data rarely conform to assumptions of independence due to phylogeny [@], space [@@diggle1998model], time [@] or other categorical variables [@bolker2009generalized].
This issue of autocorrelation is largely underappreciated in the machine learning literature and only recently and rarely have random effects been explicitely built into typical machine learning models [@eo2014tree, @hajjem2014mixed, @hajjem2017generalized, @miller2017gradient].
Most machine learning models make some assumption of independence and certainly estimates of out-of-sample predictive ability are biased if cross-validation is used without accounting for autocorrelation.
There are however a number of strategies to mitigate biases caused by autocorrelation and for gaining insight into the random effects themselves.
These include simple methods such as using random effects as normal covariates or preprocessing the data to remove autocorrelation [@].
Further methods include the creation of new covariates that encode the autocorrelation in more useful ways,
 stratified cross-validation [@le2014spatial] or using a mixed model to "stack" multiple machine learning models post-hoc [@bhatt2017improved].
These methods will be examined in more detail in the body of the review.

8. plan for the paper

<!--
in this review we will use pantheria as an example.
we will fit three models with differing degrees of interpretability.

- pantheria
- comparison a priori selection and regression
- overview of interpretable machine learning
- r squared baseline as new method
- clustering of ice plots as new method

--->

In this review I will present an illustrative analysis on the PanTHERIA dataset [@jones2009pantheria] which contains life history traits for many mammals.
I fitted four models, with variations, that span the range of interpretable.
As a point of comparison I fitted a typical model used by biologists; a simple linear model with a priori variable selection.
I also fitted a parametric statistical model, a non-parametric statistical model and a non-parametric non-statistical model.
For each of these models I demonstrate how they can be interpreted.
The full analysis is included as a reproducible R [@R] script that reads data directly from online repositories.

## Example analysis

### Data
The PanTHERIA database is a dataset of mammalian life history traits collected from the published literature  [@jones2009pantheria].
Overall it contains Todo species and information on Todo traits, complimented by a further Todo variables calculated from IUCN shapefiles for each species and remotely sensed data.
There is large amounts of missing data for many of the life history traits.
Furthermore, due to each data row being a species, the data are not independent, with species with more recent common ancestors being more likely to share life history traits.
While methods for analysing this type of data vary [@gay2014parasite, others], the cornerstone in most analyses would be phylogenetic regression that uses a seperately estimated phylogeny, converted to a covariance matrix, and included as a random effect [@magnusson2017glmmtmb, @caper].
In this illustrative analysis I will use use this dataset to examine potential factors relating to the log of the average litter size (plus one to deal with zeroes).

### Model fitting

<!--
3 models via caret [@caret]
typical ecological modelling

regularised regression

 gp [@rasmussen2004gaussian]

random forest [@wright2015ranger, @breiman2001random]

no reason effects for now. assuming iid.
-->

#### A priori variable selection

The standard approach for modelling in ecology and comparative biology is to carefully select a relatively small suite of covariates based on \emph{a priori} knowledge of the system.
As a model to commpare to, I fitted a linear model with \emph{a priori} variable selection.
I chose body size [@leutenegger1979evolution, @tuomi1980mammalian], gestation length [@okkens1993influence, @bielby2007fast], metabolic rate [@white2004does], litters per year [@white2004does] and longevity [@wilkinson2002life, @zammuto1986life].
While a specialist in 


2 or 3 questions.

### Global properties
  - gain some understanding of a system

     - predictability
        - overall how good is prediction?

    ![Predicted vs observed for the a priori selected model](figs/a_priori_var_selection-1.pdf "Predicted vs observed for the a priori selected model.")

    ![Predicted vs observed for the a priori selected model](figs/elastic_net-2.pdf "Predicted vs observed for the a priori selected model.")

    ![Predicted vs observed for the Gaussian process model](figs/gp?-2.pdf "Predicted vs observed for the a priori selected model.")

    ![Predicted vs observed for the Random Forest model](figs/ranger-2.pdf "Predicted vs observed for the a priori selected model.")


     - complexity
        - look at mtry, coefs retained, sigma
        - sigma is 0.04 = lengthscale of 0.0032? But this is in multidimensional space I think.

    ![Hyperparameter selection for the elastic net model](figs/elastic_net-1.pdf "Hyperparameter selection for the elastic net model.")

    ![Hyperparameter selection for the Gaussian proces model](figs/gp?-1.pdf "Hyperparameter selection for the Gaussian proces model.")

    ![Hyperparameter selection for the Random Forest model](figs/ranger-1.pdf "Hyperparameter selection for the Random Forest model.")

     - r2
        - compare R2 of apriri model

    - fit simpler model iml package

### Variable level properties

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

   ![PDP plot for Gestation Length in the Gaussian process model.](figs/pdp_gest-1.pdf "PDP plot for Gestation Length in the Gaussian process model.")

   ![PDP plot for Gestation Length in the Random Forest model.](figs/pdp_gest-2.pdf "PDP plot for Gestation Length in the Random Forest model.")

   ![2D PDP plot for Gestation Length and PET in the Gaussian process model.](figs/pdp_gest_pet-1.pdf "2D PDP plot for Gestation Length and PET in the Gaussian process model.")

   ![2D PDP plot for Gestation Length and PET in the Random Forest model.](figs/pdp_gest_pet-2.pdf "2D PDP plot for Gestation Length and PET in the Random Forest model.")





   ![ICE plot for Gestation Length in the Gaussian process model.](figs/ice-1.pdf "ICE plot for Gestation Length in the Gaussian process model.")

   ![ICE plot for Gestation Length in the Random Forest model.](figs/ice-2.pdf "ICE plot for Gestation Length in the Random Forest model.")

   - clustered ice
     - method description
     - results

   
   ![Clustered ICE plot for latitude in the Gaussian process model.](figs/clustered_ice_lat-1.pdf "Clustered ICE plot for latitude in the Gaussian process model.")

   ![Clustered ICE plot for latitude in the Random Forest model.](figs/clustered_ice_lat-2.pdf "Clustered ICE plot for latitude in the Random Forest model.")

### Handling non-independent data

The PanTHERIA data is an example of data that strongly violates assumptions of independently sampled data.
The autocorrelation here arises due to common ancestors of species; two species that recently diverged from a common ancestor are likely to be more similar than species whose common ancestor is in the deep past.
This autocorrelation is typically handled with a phylogenetic random effect while other sources of autocorrelation such as time or space can be similarly handled with appropriate reason effects.
The most commonly used random effect in ecological and evolutionary analyses is categorical random effects that can be used to model a wide variety of sources of autocorrelation such as multiple samples from a single individual, site or lab for example.
The naïve approach to including random effects within machine learning models would be to simply include them as covariates: categorical fixed effects as normal categorical covariates, space or time as continuous variables for example.
However to understand when this approach is or is not appropriate, we have to examine three factors as to why these effects are not just included as fixed effects in typical mixed effects models.

Firstly, we expect to extrapolate continuous random effects and expect unseen categories in categorical random effects.
Many machine learning models extrapolate poorly, for example tree based models will extrapolate in a flat line from the prediction at the extreme of the data range. 
For an effect such as space this is undesirable and we would instead typically wish the spatial prediction to return to the mean of the data.
Similarly, a categorical variable would often be encoded as a one-hot dummy variable and unseen categories would be implicitly predicted using the fitted value for the first category.

Secondly, we often have many categories and little data per category in a categorical random effect and wish to share power across groups.
This sharing of power might apply to the slope coefficients in a linear model (a random effects model with random slopes).
In both cases this can be reframed as a regularisation problem. 
For many categories with few data points per category we are simply estimating a lot of parameters and need to regularised accordingly.
Similarly, a random slopes model is fitting the interaction term between a continuous variable and a categorical variable with many categories.
Again, this is a regularisation problem.
This framing of random effects as a regularisation problem can be seen explicitely in the Bayesian formulation of random effects models using hierarchical models [@].

Finally, random effects are often included as a way to control for autocorrelation raster than being part of the expected predicted model.
For example, if all future predictions are to be for unseen categories of a categorical random effect or if all spatial predictions are to be made far from data, then we might want to construct our model simply so that the model is unbiased be these effects rather than using them directly in predictions.
Similarly if the data collection was by biased with respect to a random effect, we might want to control for this without wanting to use this effect in predictions.
For example, if data was collected by different labs or with different protocols, we might want to control for this effect but then predict the latent effect.
If the presence of a species is measured using different methods (camera trapping, visual surveys etc.) we might want to control for this, but we aim to predict the latent state "species presence", not "species presence as measured by camera trapping".
While this relate to the first point on predicting outside the range of the data, the methods for handling it can be different.

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

      - Non random CV.
      - new approach 2, fit model then mixed effects. [@bhatt2017improved]

     
### Data point level properties

  - understand individual points
    - lime [@lime, @ribeiro2016should, @lundberg2017unified , @ribeiro2016nothing]
    - explain
    - results



   ![LIME analysis of predictions of five points from the Gaussian process model.](figs/lime-2.pdf "LIME analysis of predictions of five points from the Gaussian process model..")

   ![LIME analysis of predictions of five points from the Random Forest model.](figs/lime-3.pdf "LIME analysis of predictions of five points from the Random Forest model.")


### other models

- rrf
- monotonic constraints
- [@ryo2017statistically] conditional trees (stats test before split), model based trees (linear model in the tree or something?) and bootstrap tests of significance for random forest variable selection. not great but worth mentioning.

## Future directions and conclusions

- broad encorporation into ecology
- clear distinction between predictions and inference
- suggest workflow of split, explore, test.
- viz of surface
- Bayesian bootstrap, boosted mech models.
- human experiments of interprability [@bastani2017interpreting].







