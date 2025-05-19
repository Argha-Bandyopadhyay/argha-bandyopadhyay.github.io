---
layout: post
title:  "are we sure about PCA?"
subtitle: "a neuroscientific perspective on Bayesian matrix factorization"
date:   2025-05-18
categories: explainer
---
# why are we talking about this?
These days, systems neuroscientists usually record from huge amounts of neurons in order to characterize the neural codes which guide complex behavior [(Urai et al., 2023)][urai]. However, that leads to a practical problem: how do we figure out what large groups of neurons are saying in the first place? In order to make this problem tractable/interpretable/approachable, we assume that large groups of neurons are saying something *together* - in other words, the activity of individual neurons is less relevant than the **population code** [(Yu and Cunningham, 2014)][yu and cunningham]. We often visualize this in neural state space, where each point tells us what the neural ensemble is doing at a particular point in time \\(t\\):

<figure>
  <img src="{{site.url}}/assets/figures/matrixfactorization/statespace.gif"/>
  <figcaption>Figure 1: Example of neural state space (right) for 2 example neurons with sinusoidal firing rates with a 180-degree phase offset (left). Note how both neurons' progression in time can be captured by movement along a 1D line (right). Now imagine trying to visualize this for 200 noisy neurons instead!</figcaption>
</figure>

However, this visualization quickly becomes unruly when the population of neurons we're observing has a number of neurons \\(N > 3\\). Additionally, neurons (like any other biological system) are quite noisy, meaning they frequently engage in side chatter which (we think) is mostly irrelevant to the population's overall message. These factors can make interpretation of raw neural state spaces quite difficult. 

# so what do we do?
To handle this, the first step in many analysis pipelines is **dimensionality reduction**. Dimensionality reduction is a technique which simultaneously *decreases* the amount of numbers we use to describe our data while *minimizing* how much of our data's "vibe" (technical term) we lose by using less numbers. The classical technique for this is **Principal Components Analysis (PCA)**. If not obvious from my description of dimensionality reduction, I'm going to be using the least-squares formulation of PCA - if you're unfamiliar, [better bloggers than me][alex pca blog] have written great introductions to the topic. 

<figure>
  <img src="{{site.url}}/assets/figures/matrixfactorization/pca_schematic.png"/>
  <figcaption>Figure 2: Schematic of PCA. High-dimensional complicated neural data <b>Y</b> (left) is approximated by a tall, skinny matrix <b>W</b> as per-neuron components and a short, fat matrix <b>H</b> as per-timebin components. </figcaption>
</figure>

Formally, given a data matrix \\(\mathbf{Y} \in \mathbb{R}^{N \times T}\\) with \\(N\\) neurons and \\(T\\) timebins, we're looking to break \\(\mathbf{Y}\\) into a tall, skinny matrix \\(\mathbf{W} \in \mathbb{R}^{N \times D}\\) and a short, fat matrix \\(\mathbf{H} \in \mathbb{R}^{D \times T}\\) such that:
\\begin{equation} \label{eq: lsq}
\underset{\mathbf{W, H}}{\operatorname{minimize}}{\lVert\mathbf{Y} - \mathbf{WH}\rVert_{F}^{2}}
\\end{equation}

In a nutshell, this means that we're trying to pick \\(\mathbf{W}\\) and \\(\mathbf{H}\\) so that our prediction \\(\mathbf{\hat{Y}} = \mathbf{WH}\\) is as close to \\(\mathbf{Y}\\) as possible. Because we pick \\(D \ll N\\) and \\(D \ll T\\), PCA attempts to *linearly compress* our data into per-neuron components \\(\mathbf{W}\\) and per-timebin components \\(\mathbf{H}\\) while minimizing the loss (equation \ref{eq: lsq}) of that compression. This introduces a general **low-rank matrix factorization** framework: given a complicated, high-dimensional data matrix \\(\mathbf{Y}\\), we identify the best \\(\mathbf{W}\\) and \\(\mathbf{H}\\) where quality is defined by some loss function \\(\mathcal{L}\\) which can incorporate constraints/regularization on \\(\mathbf{W}\\) and \\(\mathbf{H}\\). Other than PCA, what else falls into this framework?
- Non-Negative Matrix Factorization (NMF): same loss function as PCA, but with the constraint that \\(\mathbf{W}, \mathbf{H} \geq 0\\).
- \\(k\\)-Means Clustering: same loss function as PCA, but with the constraint that \\(D = k\\) and that the rows of \\(\mathbf{W}\\) (or the columns of \\(\mathbf{H}\\)) must have exactly one non-zero entry which \\(= 1\\).
- and [the rest of the zoo!][low rank julia] All you need is a loss function \\(\mathcal{L}\\), regularizers for \\(\mathbf{W}\\) and \\(\mathbf{H}\\), and a desired rank \\(D\\) to fit a low-rank matrix factorization model.

# this sounds great, what's the problem?
It is great! This is a powerful, flexible class of models that can be quickly fit using the machinery of linear algebra and convex optimization. But do you remember where this blog started, talking about recording from a bunch of actual neurons? The world of matrix factorization feels far-removed from the realities of experimental neuroscience, where there are a few practical concerns:

**Concern 1**: Like we discussed earlier, neurons can be really noisy! Critically, low-rank matrix factorization models do not distinguish between the concepts of signal and noise. They simply fit dimensions which capture as much of the data as possible. This complicates a main practical use for dimensionality reduction: denoising data. Many people will fit PCA models and run the rest of their analysis on this \\(D\\)-dimensional dataset with the assumption that this inherently captures the neural activity that is most meaningful for their task and higher dimensions are simply noise. However, this assumes that the task-relevant neural signal has a larger scale than the task-irrelevant neural noise - an assumption that may not be justified given the large effects of general movement/arousal signals in coordinating neural activity [(Stringer et al., 2019)][stringer].

<figure>
  <img src="{{site.url}}/assets/figures/matrixfactorization/stringer_fig.png"/>
  <figcaption>Figure 3: The first PC of neural activity in visual cortex correlates strongly with spontaneous behavior such as movement/whisking/arousal. Figure 1F of Stringer et al., 2019.</figcaption>
</figure>

**Concern 2**: It's difficult to use these models out-of-the-box on (low-to-medium firing rate) spiking data. Because there are periods of time with no spikes for each neuron, it is often difficult for these models to find accurate low-rank approximations. Therefore, it is typical to smooth spikes prior to fitting these models, with the degree of smoothing arbitrarily picked in order to ensure that these models can find a low-rank approximation of the data - this should make you feel icky inside if your goal is to build intuition for how neural systems work. Utilizing smoothness regularization on the identified temporal factors \\(\mathbf{H}\\) is a more principled approach to this, but often requires cross-validation strategies to identify the appropriate degree of regularization.

<figure>
  <img src="{{site.url}}/assets/figures/matrixfactorization/spiky_statespace.gif"/>
  <figcaption>Figure 4: Example of neural state space (right) for same 2 example neurons as Figure 1, but now with Poisson-drawn spike counts in each timebin (left). The activity is much more complicated than motion along a 1D line (right), even though we know that's the true underlying mechanism. This is a problem!</figcaption>
</figure>

**Concern 3**: Speaking of cross-validation, [it can get quite convoluted][alex cv blog] for these models. This is because these models aren't naturally built to handle missing data - the nice tools of linear algebra and convex optimization which make fitting these models relatively easy become less applicable in the setting of missing entries in the data matrix \\(\mathbf{Y}\\). Therefore, cross-validation utilizing multiple train-test splits can be prohibitively expensive for these models given the increasing size of neural datasets.

To overcome these limitations, we'd want an approach which can still leverage the general low-rank matrix factorization framework but is designed to handle each of these practical concerns. Namely, we want something which has a signal and noise model, can naturally handle spiking data, and is robust to missing data. 

# does something like that exist?
Based on those desired model characteristics (or more likely, the subtitle of this blog post), you've probably figured out that we're gonna talk about **Bayesian methods**. For those unaware, a very, very abridged tl;dr is that Bayesian methods tend to view problems through a probabilistic lens rather than an optimization framework. A longer introduction is out of the scope of this blog post, but you can find some [great intros][bayesian stats] online! For our purposes, this means that instead of getting out the single best \\(\mathbf{W}\\) and \\(\mathbf{H}\\), we're going to try to infer a posterior distribution over *all possible* \\(\mathbf{W}\\)s and \\(\mathbf{H}\\)s given the observed data \\(\mathbf{Y}\\) - we're going to call that \\(P(\mathbf{W, H} | \mathbf{Y})\\). To get this posterior, we need to specify two things: a prior distribution \\(P(\mathbf{W, H})\\) which defines what we'd expect of our parameters \\(\mathbf{W}\\) and \\(\mathbf{H}\\) without any observed data, as well as a likelihood function \\(P(\mathbf{Y}|\mathbf{W, H})\\) which relates the observed data \\(\mathbf{Y}\\) to a given set of parameters. Then, using Bayes' rule to relate conditional probabilities, we can see that:
 \\begin{equation} \label{eq: posterior}
 P(\mathbf{W, H} | \mathbf{Y}) \propto P(\mathbf{W, H})P(\mathbf{Y}|\mathbf{W, H})
 \\end{equation}
where the proportionality is up to some normalizing constant \\(Z = P(\mathbf{Y})\\) that divides the right-hand side to ensure that the posterior will integrate to 1. The power of the Bayesian framework is that the prior and likelihood can be relatively simple to compute independently, but their product can form complicated posterior distributions. There are natural parallels between this approach and an optimization framework. The prior distribution is analogous to the regularization/constraints on \\(\mathbf{W}\\) and \\(\mathbf{H}\\), while the likelihood function is similar to the unconstrained loss function. Therefore, the process of minimizing the full loss function \\(\mathcal{L}\\) is (roughly) equivalent to finding the maximum of the posterior distribution \\(P(\mathbf{W, H} | \mathbf{Y})\\), or the *single most probable* set of parameters given the observed data.

So how does that get us closer to our ideal model? Simply viewing PCA through a probabilistic lens (without even thinking about a full Bayesian model) can get us quite far. The least-squares loss function in equation \ref{eq: lsq} implies a Gaussian likelihood function where all of the dimensions have independent and equally scaled variances and the mean is linearly defined by our parameters \\(\mathbf{W}\\) and \\(\mathbf{H}\\) - this implementation of **probabilistic PCA** learns both the variance and the mean associated with this Gaussian likelihood and is naturally robust to missing data [(Tipping and Bishop, 1999)][tipping and bishop]. By assuming a non-informative prior over the parameters \\(\mathbf{W}\\) and \\(\mathbf{H}\\), you can even leverage the tools of Bayesian model selection to optimally choose the rank \\(D\\) of your approximation - intuitively, Bayesian model selection will naturally implement an *[Occam's Razor][occam]* approach, which will choose the simplest possible model that has the best evidence given the observed data [(Minka, 2000)][minka]. A closely related variant of this model is **factor analysis** which similarly utilizes a Gaussian likelihood, but learns an independent variance for each dimension rather than assuming equal variances across all dimensions: this nicely divides the problem into a 'signal' component captured by the low-rank latent factors which tries to explain the *covariance* between all of the dimensions, and per-dimension 'noise' which captures all of the remaining variance. Probabilistic PCA and factor analysis alone can handle points 1 and 3 from above!

By introducing priors on \\(\mathbf{W}\\) and \\(\mathbf{H}\\), we can make these models appropriate for spiking neural data (point 2 from above). Namely, we can leverage the Bayesian machinery of Gaussian processes to enforce smoothness in our identified temporal factors \\(\mathbf{H}\\) and integrate this into a factor analysis approach - **Gaussian Process Factor Analysis** (GPFA) is an extremely powerful neural data analysis technique which performs dimensionality reduction *and* spike smoothing in a single, unified data-driven framework [(Yu et al., 2009)][gpfa]. GPFA identifies smoothly-varying latent factors which are linearly combined to generate each neuron's spiking activity on a trial-by-trial basis, along with additive Gaussian noise. Recent work has extended GPFA by introducing a Gaussian prior on these linear combinations, switching to a Poisson noise model, and employing techniques from the Bayesian model selection literature to determine the optimal dimensionality \\(D\\) for a GPFA model [(Jensen et al., 2021)][bgpfa]. Overall, GPFA and related models highlight the utility of Bayesian matrix factorization as a framework for analyzing noisy, high-dimensional neural data in order to understand the population codes underlying complex behavior.

# so what's left?
Despite the [multitude of discoveries][gpfa gsch] enabled by GPFA and related models, I think there's still some more to be explored in this area. For tractability/interpretability reasons, most people focus solely on the posterior mean and/or maximum of the posterior as estimates output by GPFA models. However, as mentioned earlier, the power of Bayesian methods lies in learning the *distribution* over model parameters given observed data ([there's some nuance here][bob carpenter posterior], but that's definitely outside of the scope of this blog post). Functionally, this means that there is a whole family of low-rank matrix factorizations that can explain a given neural dataset quite well, and I think we should not only keep that in mind but actively search for this distribution over factorizations! Certain directions may display high posterior variance, suggesting that there is insufficient data to constrain our models or perhaps that these dimensions are less relevant for the task we are studying. Conversely, other directions with very small posterior variance are well-captured by our models, and may display more task relevance. 

There has been increasing work in this area, as an extension to GPFA uses an approximate posterior distribution to characterize the trial-to-trial variability in precise event timing [(Duncker and Sahani, 2018)][svgpfa]. Additionally, there have been significant advancements in combining these low-rank matrix factorization methods with dynamical systems modeling, and these methods similarly leverage an approximate posterior distribution over the dynamics to disentangle trial-to-trial signal variability vs irrelevant noise ([Pandarinath et al., 2018][lfads] and [Dowling et al., 2024][xfads]). These approximate posterior distributions are often used for computational tractability, but computer scientists have developed techniques for exact posterior inference for low-rank matrix factorization [(Salakhutdinov and Mnih, 2008)][bmf]. These methods may prove useful to identify factorizations in scenarios where approximate techniques tend to fail, such as when multiple different regions of parameter space result in high-quality fits to the data (multimodality) or when 'outliers' are more common than expected by approximate distributions (heavy-tailed). Overall, a probabilistic approach to matrix factorization has already proven extremely fruitful for neuroscience, and I think we can leverage the full power of Bayesian methods to discover novel principles of neural population codes. 


[urai]: https://www.nature.com/articles/s41593-021-00980-9
[yu and cunningham]: https://www.nature.com/articles/nn.3776
[alex pca blog]: https://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
[low rank julia]: https://github.com/madeleineudell/LowRankModels.jl
[stringer]: https://www.science.org/doi/10.1126/science.aav7893
[alex cv blog]: https://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
[bayesian stats]: https://www.nature.com/articles/s43586-020-00001-2
[tipping and bishop]: https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/1467-9868.00196
[occam]: https://en.wikipedia.org/wiki/Occam%27s_razor#:~:text=In%20philosophy%2C%20Occam's%20razor%20(also,smallest%20possible%20set%20of%20elements.
[minka]: https://vismod.media.mit.edu/tech-reports/TR-514.pdf
[gpfa]: https://journals.physiology.org/doi/epdf/10.1152/jn.90941.2008
[bgpfa]: https://proceedings.neurips.cc/paper_files/paper/2021/file/58238e9ae2dd305d79c2ebc8c1883422-Paper.pdf
[gpfa gsch]: https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17349932664602469396
[bob carpenter posterior]: https://statmodeling.stat.columbia.edu/2019/03/25/mcmc-does-not-explore-posterior/
[svgpfa]: https://papers.nips.cc/paper_files/paper/2018/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf
[lfads]: https://www.nature.com/articles/s41592-018-0109-9
[xfads]: https://arxiv.org/pdf/2403.01371v2
[bmf]: https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf
[neural manifolds]: https://www.nature.com/articles/s41593-019-0555-4