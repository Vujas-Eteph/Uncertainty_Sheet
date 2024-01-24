### Template for a Flash Card
  - 1) What did the authors try to accomplish?
  - 2) What were the key elements of the approach?
  - 3) What can I use myself?
  - 4) What other references do I want to follow?


Some concepts: 
 - Homoscedasticity: Same variance for all R.V.
 - Heteroscedasticity: Different variance for the R.Vs
  
## 2023
 - **[The Unreasonable Effectiveness of Deep Evidential Regression](https://openreview.net/forum?id=6XkISnR2dWo&referrer=%5Bthe%20profile%20of%20Nis%20Meinert%5D(%2Fprofile%3Fid%3D~Nis_Meinert1)) (AAAI 2023)**
   - **1) What did the authors try to accomplish?**: 
     - Look how sound the theory behind EDL is.
   - **2) What were the key elements of the approach?**:  
     - Analyzing the loss and learning process.
   - **3) What can I use myself?**:
     - The referenes, and perhaps some of the mathematical proof ?
   - **4) What other references do I want to follow?**: 
     - Amini et al. 2020 [Evidential Deep Regression](https://arxiv.org/abs/1910.02600)
     - Normal-Inverse-Gamma distribution

## 2018 
 - :star: **[Predictive Uncertainty Estimation via Prior Networks](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html) (NeurIPS 2018)**
   - **1) What did the authors try to accomplish?**: 
     - The authors propose a novel approach to estimate the uncertainty of DNN, especially to better differentiate Out-of-Distribution (OoD) examples from In-Distribution (ID). 
     - Most notably the authors include a third type of uncertainty - ***distribution uncertainty*** - in addition to the epistemic and data uncertainty commonly used. That is why they also introduce Dirichlet Prior Networks (DPNs)
     - Why separate the uncertainty in a third category? The example given is directed towards active learning, to collect more dataset samples this distribution, if the most is mostly struggling with OOD - just include it as training samples.
   - **2) What were the key elements of the approach?**:  
     - Current approaches for DNNs to estimate uncertainty compute the entropy $H(p)$ of the predictive probability distribution. (Is a straightforward way to do it).
     - Other approaches leverage MC/Ensemble like paradigms, which approximates - very loosely - a Bayesian Neural Network (ideal) as the parameters learned for every model of the ensemble are sampled from a hidden distribution close to the true distribution and averaged together. A good visualization IMO is a mountain (the true parameters distribution), but where I can only construct some towers (*i.e.*, models) to cover the complete mountains (due to limited resources, e.g. time, GPU, model, etc...). Need to rework this visualization better. But the idea is, as I can't marginalize over the complete set of possible parameters, I take N model, and take the average predictions to approximate the marginalization, which is sound mathematically from the eq (3) of the paper.
     - The main contribution comes in chapter 3. ***Prior Networks*** (PNs). Here the goal is to learn the distributional uncertainty with a PN. More precisely, as the authors tackle multi-class classification, they have a categorical problem, hence need the network needs to model a categorical distribution. To facilitate the calculation (have a closed formed) a good conjugate prior to the categorical likelihood is the Dirichlet distribution. Hence they propose Dirichlet Prior Networks (DPNs). Here the DPN is going to lean the distribution over distribution uncertainty?? (check again). The DPN is going to learn to generate the concentration parameters $\alpha_i$ of a Dirichlet distribution. This Dirichlet distribution parameterized by the DPN is used a prior??(check again), and represents the distributional uncertainty.
     - Use the KL divergence instead of the cross-entropy, as otherwise the concentration parameters degenerate... The learning/training of is worth reading again. The main idea (from my understanding) is to have ID and OoD data and train the network with the KL divergence on both samples. ***Training the DPN***. The smoothing approach is also new, note that the $1-(k-1)\sigma$ is to have the pmf sum to 1.
     - Chapter 4, deal with different way to measure uncertainty. The authors explore 4 different means.
       - 1). The classical Entropy $H(p)$ for estimating the total uncertainty
       - 2). Using the mutual information. Only possible with MC/DE methods (not sure with DPNs read again), as we need the expected data uncertainty (expectation over the entropy of each model) and the total entropy (entropy over the expectation of the predictive probability distribution of each model), and then substract then to get the Mutual information, which gives us the epistemic uncertainty of our MC/DE, which gives how spread the MC/DE are wrt to each other.
       - 3) Similar to 2... but mutual information for DPNs
       - 4) Only for DPNs, as they can model distributional uncertainty.
   - **3) What can I use myself?**:
     - ???
   - **4) What other references do I want to follow?**: 
     - ???
