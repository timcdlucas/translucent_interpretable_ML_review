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
- how has it been used in ecology (SDM [@elith2006novel; @golding2018zoon], id [@mac2018bat; @waldchen2018machine], red list [@bland2015predicting], behaviour [@browning2018predicting].
- but it has a reputation as being black box
-->

Machine learning is a collection of techniques that focuses on making accurate predictions from data [@crisci2012review].
It differs from the broader field of statistics in two aspects: 1) the estimation of parameters that relate to the real world is less emphasised than in much of statistics and 2) the driver of the predictions are expected to be the data rather than expert opinion or careful selection of plausible mechanistic models.
High-level machine learning libraries that aid the full machine learning pipeline [@caret; @scikit; @maxent; @biomod] has made machine learning easy to use.
These techniques have therefore become popular, particularly in the fields of species distribution modelling [@maxent; @biomod; @elith2006novel; @golding2018zoon] and species identification from images or acoustic detectors [@mac2018bat; @waldchen2018machine].
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

This reputation is not totally unfounded with a number of factors making machine learning models difficult to interpret (Figure @fig:schematic).
Firstly, they are often nonparametric.
They therefore estimate nonlinear relationships between covariates and response variables which can be difficult to interpret.
Furthermore, these relationships are often not summarised in a small number of interpretable parameters as would be found in a polynomial or mechanistic model.
Parameters in machine learning models often don't come with estimates of uncertainty.
Therefore, even if a model's parameters could be interpreted, distinguishing noise from signal can be difficult.
Secondly, they often fit deep interactions between covariates [@lunetta2004screening].
Even simple, two-way interactions in linear models cause confusion [@] and deep, nonlinear interactions are difficult to visualise or understand.
Thirdly, fitting machine learning models is often stochastic [@] and sometimes fitting the same model with different starting values will give a totally different set of fitted parameters (though perhaps with similar predictive performance).
However, while interpretation of machine learning models can be difficult, there is plenty of insight to be gained by fitting and appraising these models, as will be seen in this review.

<!--
2. why we would want to interpret machine learning models
- mostly interpretable machine learning is part of model verification/robustness/regulation [@molnar; @ribeiro2016should]
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
[@molnar; @ribeiro2016should].

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
Finally it is worth noting that standard statistical models are often not as interpretable as they seem; understanding the results from a statistical model is made more difficult in the presence of colinearity between covariates or when nature's true model is not in the set of models being considered [@Simpson? @gelman? @lyddon2018nonparametric; @yao2017using].
Therefore, in some cases it might be better to fit a more predictive model and sacrifice some, but not all, interpretability.
Alternatively, it might be useful to use a highly predictive model to create hypotheses which could then be tested in a more formal statistical framework.


<!--
4. deeper overview of machine learning
- supervised, unsupervised, reinforcement
- unsupervised dimension reduction commonly used, clustering less so.
- mostly focus on supervised
-->

### An overview of machine learning

Before examining how machine learning models can be interpreted it is worth reviewing the tasks commonly performed and having an overview of the types of models used [@crisci2012review].
There are three broad tasks in machine learning: i) supervised learning, ii) unsupervised learning and iii) reinforcement learning.
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
Methods for regularisation of linear models include the LASSO and other penalties [@tibshirani1996regression; @zou2005regularization; @xu2017generalized; @fan2001variable], as used by maxent [@maxent] for example, stepwise selection [@hocking1976biometrics], or Bayesian priors putting a bias towards zero [@park2008bayesian; @liu2018bayesian; @carvalho2009handling].
ii) Non-parametric, statistical models are fitted in a formal statistical framework as above the functional form is not defined in advance. Instead, flexible curves are fitted. This group includes splines (and GAMs which combine splines and other linear terms) and Gaussian processes [@rasmussen2004gaussian].
These methods retain the principled uncertainty estimates due to being statistical.
Furthermore, while the non-parametric components are often not represented by a small number of interpretable parameters, they are often controlled by a small number of hyperparameters.
If these hyperparameters are fitted in an hierarchical framework (as is common) then they are can be interpreted with associated uncertainty.
Finally, non-statistical, non-parametric methods encompass many more algorithmic methods such as decision trees (and ensembles of trees like Random Forests [@breiman2001random] and boosted regression trees [@elith2008working; @friedman2001greedy]).
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
In most cases the task is to identify a species or individual that a human could visually identify [@waldchen2018machine; @mac2018bat] therefore there is likely nothing new in the model.
Therefore, the main reason for interpreting deep convolutional networks is for model verification and to have an additional check for predictions made with the model.
The interpretation of deep neural networks has its own, large literature [@samek2017explainable; @montavon2017methods].
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
This issue of autocorrelation is largely underappreciated in the machine learning literature and only recently and rarely have random effects been explicitely built into typical machine learning models [@eo2014tree; @hajjem2014mixed; @hajjem2017generalized; @miller2017gradient].
Most machine learning models make some assumption of independence and certainly estimates of out-of-sample predictive ability are biased if cross-validation is used without accounting for autocorrelation.
There are however a number of strategies to mitigate biases caused by autocorrelation and for gaining insight into the random effects themselves.
These include simple methods such as using random effects as normal covariates or preprocessing the data to remove autocorrelation [@].
Further methods include the creation of new covariates that encode the autocorrelation in more useful ways,
 stratified cross-validation [@le2014spatial] or using a mixed model to "stack" multiple machine learning models post-hoc [@bhatt2017improved].
These methods will be examined in more detail in the body of the review.
<!--
8. plan for the paper


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
While methods for analysing this type of data vary [@gay2014parasite, others], the cornerstone in most analyses would be phylogenetic regression that uses a seperately estimated phylogeny, converted to a covariance matrix, and included as a random effect [@magnusson2017glmmtmb; @caper].
In this illustrative analysis I will use use this dataset to examine potential factors relating to the log of the average litter size (plus one to deal with zeroes).

### Model fitting

I fitted four classes of model (with variations) to the data: a linear model with a priori variable selection, a regularised linear model, a statistical, non-parametric Gaussian process model and a non-statistical random forest model with is an ensemble of decision trees with bootstraps of the input data and a reason subset of the covariates used to fit each tree.
I used five-fold cross-validation to test model accuracy and select hyperparameters for the models with hyperparameters.
Given the very different levels of flexibility in the models, this out-of-sample test of accuracy is important and given the non-statistical nature of the random forest, statistical, within sample model comparisons such as AIC are not possible.
All models were fitted with caret [@caret] in R [@R].
One make benefit of caret is that most of the procedures presented later for interpreting the models are immediately useable with over 200 machine learning models including up-to-date implementations of various models such add xgboost, h2o and keras [@]. 

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
I chose body size [@leutenegger1979evolution; @tuomi1980mammalian], gestation length [@okkens1993influence; @bielby2007fast], metabolic rate [@white2004does], litters per year [@white2004does] and longevity [@wilkinson2002life; @zammuto1986life].
While a specialist in the field may well have chosen different variables, this is a reasonable starting point.

#### Statistical, parametric models

If we have many covariates relative to sample size and have minimal a priori knowledge of the system we may wish to include all the covariates in a linear model but regularise the coefficients.
Similarly, if we want to include many interactions or transformed variables, the number of covariates can grow rapidly and regularisation becomes vital.
This approach is also sensible if we care more about prediction than about unbiased estimates of parameters.
The simplest regularised linear models are ridge regression [@], that includes a penalty on the square of the coefficients, and LASSO [@] that penalises the absolute value of the coefficients and therefore more strongly penalises smaller values.
A common model is the elastic net that includes both the ridge penalty and the lasso penalty.
This is the model I fitted to the PanTHERIA dataset.
The total strength of the penalty, and the relative contribution of the two penalties were selected using cross-validation (figure @fing:enethyp).


#### Non-parametric, statistical models

Given the parametric nature of the elastic net model, the way to include nonlinear responses and interactions is to define them manually before model fitting.
This however still imposes important restrictions as it is difficult to know which nonlinear are potential useful and the model is still ultimately constrained by the effects we can think of to include (typically polynomials, log and exponential transforms and perhaps sine transforms).
In contrast, non-parametric models like Gaussian processes [@rasmussen2004gaussian]  require no pro-specification of functional forms and instead the overall flexibility of the model is controlled with a hyperparameter.
Given their statistical nature, the uncertainty estimates around predictions are a natural part of the model and we would hope for them to be quite well calibrated even if we extrapolated far from the data.
I have fitted a Gaussian process model with a radial basis kernel [@kernlab], selecting the scale hyperparameter using cross-validation (figure @fig:gphyp).


#### Non-parametric, non-statistical models

Finally, I fitted a random forest model [@breiman2001random; @wright2015ranger] as an example of a non-statistical, non-parametric model.
Random forests tend to be easy to use with few hyperparameters and are less likely to overfit than other tree models like single decision trees or boosted regression trees.
Random forests using the ranger package via caret have three hyperparameters.
Split rule, which I kept constant at 'variance'.
The maximum number of data points at a leaf, which can be used to prevent overfitting was selected by cross-validation.
Finally, the number of randomly selected covariates to be used to build each tree was also selected by cross-validation (figure @fig:rfhyp).

2 or 3 questions.

### Global properties

<!---
  - gain some understanding of a system

     - predictability
        - overall how good is prediction?

--->

The first level of interpretation we can examine is the global level; what do the fitted models tell us about the system as a whole.
One global property of interest is how predictable the system is.
This can be assessed using scatter plots of observed versus out-of-sample predictions (figure @fig:enetpredobs @fig:gppredobs @fig:rfpredobs ) as well as metrics such as $r^2$ or the root mean squared error.
Random forests are effective here as they are fast to fit, robust and need relatively little tuning.
If a random forest has poor predictive performance then it is likely that either vital covariates are missing from the dataset or that the response is in fact very noisy.
The random forest model fitted here has fairly good predictive performance (figure @fig:rfpredobs) with an $r^2$ of 0.67.
However, it can be seen that certain species, particularly those with very large litters, are predicted quite poorly.
We can be fairly sure that this trait is not noisy as the evolutionary consequences of litter size are large.
Therefore we are probably missing some crucial covariates.

We can also use predictive performance of models like Random forest or elastic net to scale our expectations for how well a more statistical or mechanistic model fits the data.
Here, the linear model with a priori variable selection (figure @fig:aprioripredobs) had performance not much worse than the elastic net model (figure @fig:enetpredobs) but considerably worse than the random forest (figure @fig:rfpredobs).
The similarity between the elastic net model and a priori selection implies that the a priori variable selection was fairly good given the covariates available.
However, the fact that random forest performs much better than the a priori linear model suggests that there are nonlinearities or interactions that paint should have been in the a priori model.
It is important to be clear that this is not a suggestion to go back and add these variables to our a priori model. 
This would amount to quite severe data snooping and would bias any significance tests performed on the a priori model.

![Predicted vs observed for the a priori selected model](figs/a_priori_var_selection-1.pdf "Predicted vs observed for the a priori selected model."){#fig:aprioripredobs}

![Predicted vs observed for the a priori selected model](figs/elastic_net-2.pdf "Predicted vs observed for the a priori selected model."){#fig:enetpredobs}

![Predicted vs observed for the Gaussian process model](figs/gp?-2.pdf "Predicted vs observed for the a priori selected model."){#fig:gppredobs}

![Predicted vs observed for the Random Forest model](figs/ranger-2.pdf "Predicted vs observed for the a priori selected model."){#fig:rfpredobs}

<!---
     - complexity
        - look at mtry, coefs retained, sigma
        - sigma is 0.04 = lengthscale of 0.0032? But this is in multidimensional space I think.
--->

We can also attempt to interpret the hyperparameters of our models to try to understand something about the complexity of the system.
For the elastic net model, the lambda parameter and the number of non zero coefficients give us some idea of the systems complexity (figure @fig:enethyp); if very few variables are retained and we get good predictive performance this suggests a simple system.
Here we have Todo.
Similarly, the length scale, $\sigma$, in the Gaussian process model is a crude measure of complexity, with small values implying the the functional relationships are highly non linear (figure  @fig:gphyp).
Finally, the random forest model has two hyperparameters (figure @fig:rfhyp); mtry is the number of randomly selected covariates to build each tree with and min.node.size is the maximum number of datapoints that can be in a leaf node of a tree.
min.node.size protects against overfitting and gives an indication of how much noise relative to signal there is.
Here, the smallest value of min.node.size tested gets elected which implies there is not much noise in the data relative to signal.
The selected value for mtry was 20.
mtry can be difficult to interpret and depends on the number of covariates included in the model.
Very small values imply little or no interactions between covariates while intermediate or high values indicate that there are interactions between covariates.
However, large values like the 20 selected here does not indicate interaction depths of 20.
Instead it more likely implies that there are many useless covariates and so 20 covariates are needed to avoid trees with no useful covariates.
This can be examined further by fitting models with additional random covariates.


![Hyperparameter selection for the elastic net model](figs/elastic_net-1.pdf "Hyperparameter selection for the elastic net model."){#fig:enethyp}

![Hyperparameter selection for the Gaussian proces model](figs/gp?-1.pdf "Hyperparameter selection for the Gaussian proces model."){#fig:gphyp}

![Hyperparameter selection for the Random Forest model](figs/ranger-1.pdf "Hyperparameter selection for the Random Forest model."){#fig:rfhyp}

<!---
     - r2
        - compare R2 of apriri model

  
--->

  - fit simpler model iml package

### Variable level properties

The next level that we can examine is the variable level.
This can include random or fixed effects.
We can examine variable importence, importance of interactions between pairs of covariates and start to examine the functional responses of covariates.
It is important however to remember that these models are not designed for inference; the following methods sour thought of as hypothesis generation and more formal, subsequent tests (on a different dataset) would be needed to confirm relationships between covariates and the response variable.

Table todo shows the top five most important variables as determined by the three models.
These importance measures are not in absolute units so they are scaled such that the most important covariate has a value of 100.
For the regularised linear model, variable importance is given simply by the magnitude of the regression coefficients (i.e. ignoring the sign) and these raw values might be more useful than the scaled importance values.
We can see that gestation length comes top for all three models and that latitude and PET are prominent in all three as well.
Fitting multiple models and searching for consistency is one useful way to increase confidence in results.
Some models also allow tests of significance on variable importance measures (table todo).
While these come with all the normal caveats for significance testing, the scaling might be more useful interpretation than the earlier values scaled by the maximum importance values.

While caret provides an easy interface to getting variable importance measures for many model types, the calculations being performed are varied.
While this review is avoiding going into too much, model specific, detail it is important to note that there are different ways of calculating variable importance for a given model and some are more correct than others.
For the random forest model the type of variable importance calculation is important and depends on the type of variable being used.
Firstly, permutation variants importance values are now reliable (though computationally slower) than other methods like Gini impurity [@].
Secondly, in the presence of a mix of continuous and categorical covariates, all methods performed on standard random forests are biased towards selecting continuous covariates.
If accurate variable importance measures are needed, a related model, conditional inference forests, should be used instead [@].
This is not required here because the covariates are all continuous.

It is also worth noting that the reliability of variable importence measures differs between model types.
For example, repeatedly fitting a neural network to these data gives very different results each time (Figure S1todo).
In contrast, the reliability with which Gaussian processes and linear find a global maximum and the randomisation inherent in random forest means they tend to give similar results each time.



<!--- https://www.google.co.uk/url?sa=t&source=web&rct=j&url=https://www.statistik.uni-dortmund.de/useR-2008/slides/Strobl%2BZeileis.pdf&ved=2ahUKEwj1puvJm-reAhUF_KQKHWxGDlcQFjAAegQIAxAB&usg=AOvVaw0TvuGko49w5Nk8pqQ81oAD
--->

Once some important covariates have been identified, it is useful to examine the shape of the relationship between covariate and response. 
The simplest way to do this is a partial dependence plot (PDP) in which the model is evaluated at a range of values of the covariate of interest with all other covariates held at their mean (or mode for categorical variables).
All responses are linear for the regularised linear model so a PDP us not useful.
The PDPs for gestation length for the Gaussian process and random forest models are shown in figures @fig:pdpgestgp and @fig:pdpgestrf.
It can be seen that neither response is linear and are both decreasing for low values of gestation length.
However, the PDP for the Gaussian process model is increasing at high values of gestation length and is similar to a square curve.
In contrast, the random forest model is flat at high values of gestation length.

<!---
  - generate hypotheses (variable level)

    - var imp
      - what is it? 
      - easy with caret
      - results
      - ranger importance p values. 

    - interaction importance
      - iml package
      - what is?

    - ice and PDP
      - define pdp then ice
      - results
--->


![PDP plot for Gestation Length in the Gaussian process model.](figs/pdp_gest-1.pdf "PDP plot for Gestation Length in the Gaussian process model."){#fig:pdpgestgp}

![PDP plot for Gestation Length in the Random Forest model.](figs/pdp_gest-2.pdf "PDP plot for Gestation Length in the Random Forest model."){#fig:pdpgestrf}

While PDPs are evaluated at just median of the other variables, the variable importance measures calculated above are evaluated over all training data.
There can therefore be a mismatch where a PDP looks flat while the variable importance use high.
Relatedly, the PDP gives no information on interactions because it is only evaluated at one values of the other covariates.
To address these issues we can calculate the interaction importance for each covariate (table todo).
This value is given by decomposing the prediction function into contributions from just the focal covariate, contributions from everything except the focal covariate and contributions that rely on both the focal covariate and non-focal covariates together.

Once we have identified covariates with important interactions we can use individual conditional expectation (ICE) plots.
Like PDPs, ICE plots calculate the predicted response value across a range of the focal covariate.
However, instead of holding the other covariates at their median, they evaluate and plot one curve for each data point (Figure @fig:icegestgp and @fig:icegestrf).
In these plots we can start to see that the response curve differs depending on what value the other covariates take.
As the number of data points increases, these plots can get very busy and so clustering the curves is useful (figure @fig:clusticelatgp and @fig:clusticelatrf).
Here we can clearly see the range of responses that exist for a single covariate, with latitude having a positive relationship with litter size in some cases and a negative relationship in others.



![ICE plot for Gestation Length in the Gaussian process model.](figs/ice-1.pdf "ICE plot for Gestation Length in the Gaussian process model."){#fig:icegestgp}

![ICE plot for Gestation Length in the Random Forest model.](figs/ice-2.pdf "ICE plot for Gestation Length in the Random Forest model."){#fig:icegestrf}

 
![Clustered ICE plot for latitude in the Gaussian process model.](figs/clustered_ice_lat-1.pdf "Clustered ICE plot for latitude in the Gaussian process model."){#fig:clusticelatgp}

![Clustered ICE plot for latitude in the Random Forest model.](figs/clustered_ice_lat-2.pdf "Clustered ICE plot for latitude in the Random Forest model."){#fig:clusticelatrf}


Gaussian process models and random forests implicit consider very deep interactions which become increasingly difficult to interpret.
However, if we can identify important two way interactions we can start to interpret these.
We can find the interaction strength between two features in a similar fashion to finding variable importance.
We can examine the 2D PDP of two covariates (figure @fig:2dgestlatgp @fig2dgestlatrf) and calculate what proportion of the curve is explained by the sum of the two 1D PDPs (e.g. figure @fig:pdpgestgp).
We can therefore take one covariate that we know has strong interactions (Todo as seen in table todo) and calculate the two-way interaction strength between that covariate and all other covariates (table todo).
Finally, once important interactions have been identified, the 2D PDP can be examined to determine the shape of that interaction (figure @fig:2dgestlatgp and @fig:2dgestlatrf).
Looking at the 2D PDP of gestation length and latitude for the random forest model we can see that something.


![2D PDP plot for Gestation Length and PET in the Gaussian process model.](figs/pdp_gest_pet-1.pdf "2D PDP plot for Gestation Length and PET in the Gaussian process model."){#fig:2dgestlatgp}


![2D PDP plot for Gestation Length and PET in the Random Forest model.](figs/pdp_gest_pet-2.pdf "2D PDP plot for Gestation Length and PET in the Random Forest model."){#fig:2dgestlatrf}





<!---
   - clustered ice
     - method description
     - results
--->
  
### Handling non-independent data

The PanTHERIA data is an example of data that strongly violates assumptions of independently sampled data.
The autocorrelation here arises due to common ancestors of species; two species that recently diverged from a common ancestor are likely to be more similar than species whose common ancestor is in the deep past.
This autocorrelation is typically handled with a phylogenetic random effect while other sources of autocorrelation such as time or space can be similarly handled with an appropriate random effects.
The most commonly used random effect in ecological and evolutionary analyses is categorical random effects that can be used to model a wide variety of sources of autocorrelation such as multiple samples from a single individual, site or lab for example.

Given the types of machine learning discussed in the introduction, we can see that including random effects with parametric or non-parametric statistical models is entirely seasons with flexible modelling packages [@stan; @inla].
However including random effects with non-parametric, non-statistical models is difficult.
While these models are starting to be developed [@hajjem2014mixed; @hajjem2017generalized; @eo2014tree; @miller2017gradient;@REEMtree], they are not available on R packages and are only implemented for a small subset of machine learning algorithms and don't necessarily benefit from the computational improvements implemented in the most up-to-date packages [@wright2015ranger; @xgboost].
Therefore, generic methods for handling random effects, that can be used with any machine learning algorithm, are useful.
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

Finally, random effects are often included as a way to control for autocorrelation rather than being part of the expected predicted model.
For example, if all future predictions are to be for unseen categories of a categorical random effect or if all spatial predictions are to be made far from data, then we might want to construct our model simply so that the model is unbiased be these effects rather than using them directly in predictions.
Similarly if the data collection was by biased with respect to a random effect, we might want to control for this without wanting to use this effect in predictions.
For example, if data was collected by different labs or with different protocols, we might want to control for this effect but then predict the latent effect.
If the presence of a species is measured using different methods (camera trapping, visual surveys etc.) we might want to control for this, but we aim to predict the latent state "species presence", not "species presence as measured by camera trapping".
While this relate to the first point on predicting outside the range of the data, the methods for handling it can be different.

Given these issues we can consider how to include random effects into machine learning models and then examine the results when these are applied to the PanTHERIA analysis.
As discussed above, the phylogenetic effect is the clearest in the PanTHERIA dataset.
One way of including phylogenetic information in an analysis is to treat a taxonomic level such as genus as a categorical reason effect.
While this is less principled than properly including the phylogeny, it is simple and makes it possible to demonstrate categorical random effects with the PanTHERIA database.

First considering the case of using genus as a categorical random effect to encapsulate some phylogenetic information, the first issue is that by default, new genera will have predictions that implicitly they are of the reference genus in the one-hot encoded dummy variables. 
Instead we can give every genus it's own dummy variable.
While this would cause identifiability with the intercept in a linear model, the random columns and greedy splitting of random forests means this will not cause problems.
The second issue above was that of regularisation.
Random Forest will automatically consider all interactions between our covariates and genus effect.
Random Forest is natively regularised by the bootstrap aggregation, and the complexity of the model is further controlled by hyper parameters as in figure xx.
The new model can therefore be fitted in the same way as the old model.
However, given that I have added many covariates, I increased the range of the mtry parameter.
A alternative approach available in ranger (but not in randomForest) is to weight columns.
I also ran a model where the probability of including one of the genus columns was scaled by the number of genera.
The final consideration above was the case where we expect all predictions to be made on new categories.
This does not particularly apply in this analysis because we may well want to make predictions for a species not in the dataset but whose genus is.
Furthermore, in the case of random forest, the above methods are suitable even if this wasn't the case.
However, given a model that cannot regularise as effectively, we might want to control for genus without including it as a covariate in the model.
In this case we can simply weight the data so that each genus is equally represented or so that each genus represented proportionally to three number of species in each genus in the full prediction set, which could be for example all mammals.
Many models in caret accept a weight argument so this is a very general solution.

If however, we wish to include the full phylogeny in our model, we need different methods.
The first method is to include all the phylogenetic information in covariates [@hengl2018random].
Given the data set of todo datapoints we can do this by defining Todo new covariates that measure the phylogenetic distance between datapoints.
That is, the first new covariate is the phylogenetic distance between every datapoint and the first datapoint, then this is repeated to create todo new covariates.
This method is relatively untested but is general and can work with any machine learning algorithm.
However, interpretable of the strength of the phylogeny will be relatively difficult as it is encoded as Todo different covariates.

The second method involves fitting multiple machine learning models and then using phylogenetic regression to 'stack' them. 
We fit a number of machine learning algorithms and make out of sample predictions within the cross-validation framework.
We then fit a phylogenetic mixed-effects model using the out-of-sample predictions as covariates and constraining the regression coefficients to be positive.
This method is likely to be very effective at prediction and the phylogenetic component of the regression is interpretable as it would be in any normal phylogenetic regression.
However, this method only corrects for the biases from autocorrelated data after the fact; while it may still be possible to interpret the machine learning models as we have done previously, the computed nonlinear relationships remain biased.


While I cannot demonstrate the handling of spatial or temporal autocorrelation with this dataset it is worth some brief discussion.
Spatial random effects can be handled in the same ways as the phylogenetic effects, in fact both of the methods proposed come from the spatial literature [@hengl2018random, @bhatt2017improved].
Temporal effects are easier to handle as they are one dimensional with causation only able to occur in one direction.
Furthermore, they have been studied in detail in the machine learning literature [@jeong2008non].
For regular time series we can typically include covariates created from the lagged response variable while for irregular time series we can create covariates like "mean response within X units of time previous to this datapoint".

<!---
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

--->     


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

<!---

- broad encorporation into ecology
- clear distinction between predictions and inference
- suggest workflow of split, explore, test.
- viz of surface
- Bayesian bootstrap, boosted mech models.
- human experiments of interprability [@bastani2017interpreting].

--->

It is clear that machine learning continues to grow as a tool in ecology.
However, it currently remains used almost solely for purely predictive purposes.
The next stage is for machine learning methods to find their place within other statistical tasks undertaken by ecologists.
One important step for this to occur is for ecologists to be more clear about the purposes of their analyses; is a well defined hypothesis being tested, is a dataset being employed for potential relationships to drive hypothesis generation, it is prediction the main focus.
This clarity makes it possible to be clear about the trade-offs in any statistical analysis and to use the most effective tools given the desired outcomes.
Using simple linear models is often not optimal if discovery of relationships or predictions are the aim; if a formal hypothesis is being tested random forests are unlikely to be the best choice.
Finally, being clear about the aims allows sensible planning on how data will be used in the longer term.
If the aim is to discover some relationships and then formally test them, the best use of a given dataset may be too split it and use half for disovery and half for hypothesis testing.
This workflow would not occur to an analyst who was unclear about their task.





