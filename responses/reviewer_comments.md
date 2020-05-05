


Reviewer(s)' Comments to Author:

Reviewer: 1

Comments to the Author
The review by Tim Lucas aims to ‘demystify’ the widely perceived black box aspect of machine learning models, by accompanying the reader along the analytical steps using a real dataset. I found the article compelling and convincing, very well written and well structured, and noteworthy, valuable for both readers not so familiar with ML but also giving valuable ‘tips’ for more advanced users.

_Thank you_

By focusing on the interpretability of models in a supervised problem and explicitly splitting the possible aims (making good predictions and hypothesis generation), we can skip all the more general introduction that is given somewhere else in the literature, i.e. the Crisci et al. 2012 review that is cited for instance. Though I could recommend the following ref for a broad introduction: Domingos, P. (2012) A few useful things to know about machine learning. Commun. ACM 55, 78.

_will add_

I think a reference for beginners for the rather ML-specific terminology (features, hyperparameter, k-fold cross-validation…) may be added somewhere also, but unfortunately cannot think of a good one (most of this comes along reading papers and many blog posts…), could a box be added within the article maybe?

_yes will add a table or box. and make sure I use one column throughout_ job

One general comment I could make is that I missed something about the problem of extreme values, that are tricky to predict (see comment below on L.198). Of course, extremes are by definition harder to predict, but in my experience of ML (for highly dimensional microbial datasets), random forest fails (sometimes largely) to accurately predict extremes even though it performs better overall than neural nets for instance. Yet neural nets somehow better predict those (using cross-validation), but at the cost of larger residuals for samples in-between (not clear if I overfit or not…). But feel free not to account for this comment, this is somewhat beside the point of your review.

_this is a big area so I can't do fully. but I agree that its important so will add a caution. also it's an important part of interpretation so will add some specifics. I also added a sentence in the caption of figure 2 pointing out that the Random Forest model cannot extrapolate beyond the range of the data._

L.120: Not convinced by this assertion, it all depends of the considered functional traits and the organizational level, but I understand the point made here for the review purpose… For instance, for microbial taxa that I work with, monophyletic groups can encompass highly variable trophic strategies (parasites, mutualists, autotrophs, heterotrophs, mixotrophy…) as well as cell size (sometimes few orders of magnitude). But I would more agree for large size organisms such as mammals I guess, though overall the phylogenetic signal in traits conservatism remain controversial. Again, this is beside the point of the review..

_I have slightly altered the sentence. "species with more recent common ancestors may have more similar life history traits". As you state there are cases when this isn't the case but we can't necessarily know until we've measured the dependence on phylogeny._

L.198 and Figure 2: You may remind in the figure caption that you model the litter size and also briefly comment the results. Also, this low predictability for extremes values is true whichever the model used here (figure 2) and actually a common problem. Could you develop a bit this somewhere in the ms?

_yes will do_

L.205: To me, the linear model (2a) seem almost unable to model the litter size, and an R2 of 0.34 does not compete much with the 0.53 obtained with elastic net.

L.306: You may add the following refs about random forest and interactions discovery:
Basu et al., 2018, Iterative random forests to discover predictive and stable high-order interactions, PNAS
Wright et al. 2016, Do little interactions get lost in dark random forests? BMC Bioinfo

_will do_


Typos that I could find:
L.8: The sentence “While it can be difficult to interpret fitted machine learning models these models can, with careful examination, be used to inform our understanding of the world” reads weird.
L.12: “In this review, I review”
L.117: Double “use”
L.341: “This fact is supported by the fact”
L.536: “need to BE done”
_Thank you, all these errors have been corrected._



Reviewer: 2

Comments to the Author
This is a useful paper comparing traditional statistical techniques used in ecology to machine-learning methods, with a focus on methods for visualization and interpretation that can be used across methods. It brings together approaches rarely shown by side by side in a way I haven't seen before, and I believe it will be useful to many ecologists.

_thank you_

The author has a considerable challenge in harmonizing methods that come from different fields that have confusing parallel vocabularies and concepts. Readers are likely to have familiarity with some of the approaches but not all, and have different interpretations of similar concepts.  For this reason I think the paper could be improved with more attention to definitions, structure, and consistency to make it easier to follow and hang together conceptually.  At times it reads like a smorgasbord of model-interrogation methods. I note that the author does attempt to address all the following issues at some level, but as the paper is primarily meant to be explanatory, the bar for coherence and consistency is high.  For this reason I am being more prescriptive about language and rhetoric than I would be in a typical review.
_I agree that the bar for language should be high. Thank you for all the useful suggestions.

Structurally, the distinction between overall model performance interpretation, variable-level interpretation, and point-level of interepretation is a great way to lay out this paper.  It would be stronger if this trinity were expanded more in the introduction and used in the discussion/conclusion to frame the paper. Repetition is your friend.

_Thank you appreciating this way of structuring the paper. I have emphasised the three levels + autocorrelation structure in the abstract (lines xx), towards the beginning of the introduction (lines xx) and in the conclusion lines(xx). I have also added a sentence to give a brief overview of which levels of interpretation might be important for model verification (lines xx) and slightly edited the subsequent paragraph (lines xx) to more closely reflect the global, variable, individual structure. I have also edited the first sentence in the variable level properties to reinforce that this is the second (of three) levels to interpret._

For this reason, I'd also suggest moving the discussion of random effects into a separate section outside of these three.  While related to variable-level interpretation, it's effectively a whole other topic, and it breaks up the flow being in the middle.

_I have moved the data-point level properties section before the section on handling non-independant data. I have made a few small edits so that the rest of the manuscript reflects this change (for example separate variable and autocorrelation in lines 109-110xx)._

There are a number of places where laying out clear definitions for terms as the author uses them would be of help. The paper refers to "statistical" vs. "non-statistical" models but I don't think adequately describes what these mean. Box 1 goes into this somewhat, with the phrase "They are statistical because they will include some kind of likelihood function that makes the model probabilistic."  I think this is incomplete and I think the author should go into the difference in the main text to provide more clarity. My sense is that the author defines a "statistical" model as having a likelihood, being fit so as to maximize that likelihood or estimate its distribution, and having estimates of uncertainty derived from the distribution of likelihood values, as well as prediction uncertainty derived from the probabilistic disribution of error terms.  In contrast, "a non-statistical" model may be fit using an objective function other than likelihood, usually of predictive performance, and does not have uncertainty estimates based on likelihood but may have them derived from other methods (e.g., bootstrapping).  Since this isn't a division that has wide consensus, I suggest using language indicating where definitions are the authors', e.g., "In this paper, I define _statistical_ as..."
_I have added clarification on statistical models in lines xx-xx. I have also highlighted that these definitions are working definitions for the paper in line xx, xx and box 1. I have added a few further details in box 1._

Other terms may be familiar to some ecologists and not others, and would benefit from more structured explanation.  For instance, regularization for statistical models is explained in in Box 1, but not for non-statistical models, where it is only mentioned in the random effects section.  It would be useful to define regularization and explain the different ways it is achieved for different models in one place.
_I have added a paragraph (lines xx-xx) that explains some of terms that are common in machine learning but rarer in more standard statistics. I have tried to be brief, but have explained the need for out-of-sample data, overfitting, and regularisation. I have added a sentence in box 1 broadly describing regularisation in nan-statistical models (though this is a very large subject). Hopefully now that the general concept of regularisation is explained, I can explain the specifics without them feeling disjointed._

The biggest definitional issue is, of course, "machine learning," and despite defining it in the first sentence of the paper, it is used loosely and somewhat confusingly. At times it seems to refer to all models used in the paper, at times only "non-statistical" models. For instance, on p.23, line 59, Does "Many machine learning models extrapolate poorly" refer to all models or some subset? There are of course many ways people use the term, so again I suggest something along the lines of "'Machine learning' has many definitions; in this paper I use the term to refer to..."
_Thank you for pointing this out. The aim was to only use machine learning as defined at the beginning. I have checked every use of the words 'machine learning' and specific non-statistical or non-parametric where needed._

The final paragraph of the introduction could better lay out the purpose of the paper.  Currently it jumps into the specific data set and models, but I think a better topic sentence would be along the lines of, "In this review I demonstrate how both traditional and machine-learning models can be interpreted and visualized so as to...", and leave the specifics of the data and models to the next section
_will do_

The "Example Analysis" section would be improved if specific problem were described prior to the "Data" subsection.

_will do_

All the figure captions are short and do not orient the reader to the plots, and in my opinion the paper text does not do so either, describing the plots only in general concept where they are referred to.  I suggest more detailed explanations in the captions.  For instance, Fig 1. could say something along the lines of "Plots of hyperparameter value vs. model predictive performance for each model.  Each plot represents the value of the hyperparameter controlling the degree of regularization for the model. A) Both the penalty term (lambda) and the Lasso/Ridge fraction control regularization in the Elastic Net Model, and performance is highest at...".   As you are introducing the reader to a possibly new visualization as well as as new models, it's prudent to err on the side of repetition.
_I have added considerable amounts of detail to all the figure captions. I have made a number of edits to tie the text to the figures more as well._


In Figures 3 and 4 (at least), I think including the linear partial effects of the Elastic Net model would help in maintaining parallel structure and driving home the differences between the models.

_I agree and have added the linear partial effects plots to figures 4 and 5. However, the clustered ICE plot does not make much sense for the linear model as after normalisation, all the curves are identical. I have therefore ommitted this plot. As implied but not demanded, I also added a plot for the elastic net model in the 2D PDP section as I agree that keeping the parallel structure of the three models is very useful._


I think software implementations are worthy of their own subsection that describes both the tools used herein and the general categories of model-implementing packages, general workflow support (e.g. caret), and the model-independent diagnostic and visualization tools such as those that make ICE and LIME plots. Mention of comparable sets of tools in different computer languages would be a useful addition, as well.

_will do_

A few other non-exhaustive small notes (page:line)
2:12 - "In this review I review" --> "Here I review"
4:61 - In discussing the use of ML for exploratory analyses, it would be prudent to mention the "garden of forking paths" problem (Gelman and Loken 2015, https://doi.org/10.1511/2014.111.460 )
8:34 - It would be prudent to mention that caret is being somewhat deprecated in favor of a different suite of packages built around *parsnip*.
14:226 - One should use full descriptive terms for these hyperparameters, not just the names used in code..
23:363: "One-hot" may not be a term understood by the audience, but I think it is easier just to remove it than to define it.
23:370 - explicitely --> explicitly

_I have corrected the language errors. I have cited the forking paths paper and Nosek et al. 2012 both in the intro (line xx) and the conclusions (lines xx). I have mentioned parsnip in the new software section (lines xx - xx)._
