Based on [Awesome Uncertainty in Deep learning](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning)

[Temperature Scaling](https://github.com/gpleiss/temperature_scaling)


## Ensemble-Methods
- [ ] ~~Weighted Ensemble Self-Supervised Learning [[ICLR2023]](<https://arxiv.org/pdf/2211.09981.pdf>)~~
- [x] Agree to Disagree: Diversity through Disagreement for Better Transferability (D-BAT)  [[ICLR2023]](<https://arxiv.org/pdf/2202.04414.pdf>) - [[PyTorch]](<https://github.com/mpagli/Agree-to-Disagree>)
- [x] Packed-Ensembles for Efficient Uncertainty Estimation [[ICLR2023]](<https://arxiv.org/abs/2210.09184>) - [[PyTorch]](<https://github.com/ENSTA-U2IS/torch-uncertainty>)
- [ ] ~~Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling [[AAAI2023]](<https://arxiv.org/abs/2302.01312>)~~
- [x] Deep Ensembles Work, But Are They Necessary? [[NeurIPS2022]](<https://arxiv.org/abs/2202.06985>)
- [x] FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation [[NeurIPS2022]](<https://arxiv.org/abs/2206.00050>)
- [x] Prune and Tune Ensembles: Low-Cost Ensemble Learning With Sparse Independent Subnetworks [[AAAI2022]](<https://arxiv.org/abs/2202.11782>)
- [ ] ~~Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity [[ICLR2022]](<https://arxiv.org/abs/2106.14568>) - [[PyTorch]](<https://github.com/VITA-Group/FreeTickets>)~~
- [ ] ~~On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection [[ECCVW2022]](<https://arxiv.org/abs/2207.07517>)~~
- [x] Robustness via Cross-Domain Ensembles [[ICCV2021]](<https://arxiv.org/abs/2103.10919>) - [[PyTorch]](<https://github.com/EPFL-VILAB/XDEnsembles>)
- [x] (MiMO) Training Independent subnetworks for robust prediction [ICLR2021](https://openreview.net/forum?id=OGg9XnKxFAH)
- [x] Masksembles for Uncertainty Estimation [[CVPR2021]](<https://nikitadurasov.github.io/projects/masksembles/>) - [[PyTorch/TensorFlow]](<https://github.com/nikitadurasov/masksembles>)
- [ ] Uncertainty Quantification and Deep Ensembles [[NeurIPS2021]](<https://openreview.net/forum?id=wg_kD_nyAF>)
- [ ] ~~Uncertainty in Gradient Boosting via Ensembles [[ICLR2021]](<https://arxiv.org/abs/2006.10562>) - [[PyTorch]](<https://github.com/yandex-research/GBDT-uncertainty>)~~
- [ ] ~~Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning [[ICLR2020]](<https://arxiv.org/abs/2002.06470>) - [[PyTorch]](<https://github.com/SamsungLabs/pytorch-ensembles>)~~
- [ ] ~~Maximizing Overall Diversity for Improved Uncertainty Estimates in Deep Ensembles [[AAAI2020]](<https://ojs.aaai.org/index.php/AAAI/article/view/5849>)~~
- [ ] Hyperparameter Ensembles for Robustness and Uncertainty Quantification [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html>) (Design )
- [ ] ~~Bayesian Deep Ensembles via the Neural Tangent Kernel [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/0b1ec366924b26fc98fa7b71a9c249cf-Abstract.html>)~~
- [x] BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning [[ICLR2020]](<https://arxiv.org/abs/2002.06715>) - [[TensorFlow]](<https://github.com/google/edward2>) - [[PyTorch]](<https://github.com/giannifranchi/LP_BNN>)
- [ ] Uncertainty in Neural Networks: Approximately Bayesian Ensembling [[AISTATS 2020]](<https://arxiv.org/abs/1810.05546>) (Might need it to take the uncertainty of the previous frame, in order to use it as prior ??)
- [ ] ~~Accurate Uncertainty Estimation and Decomposition in Ensemble Learning [[NeurIPS2019]](<https://papers.nips.cc/paper/2019/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html>)~~
- [ ] ~~Diversity with Cooperation: Ensemble Methods for Few-Shot Classification [[ICCV2019]](<https://arxiv.org/abs/1903.11341>)~~
- [ ] ~~High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach [[ICML2018]](<https://arxiv.org/abs/1802.07167>) - [[TensorFlow]](<https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals>)~~
- [ ] ~~Simple and scalable predictive uncertainty estimation using deep ensembles [[NeurIPS2017]](<https://arxiv.org/abs/1612.01474>)~~
- [ ] Diverse Lottery Tickets Boost Ensemble from a Single Pretrained Mode [[AAAI Not sure 2022]](https://openreview.net/pdf?id=rCzgE3zHL-q)


## Auxiliary-Methods/Learning-loss-distributions
- [ ] Post-hoc Uncertainty Learning using a Dirichlet Meta-Model [[AAAI2023]](<https://arxiv.org/abs/2212.07359>) - [[PyTorch]](<https://github.com/maohaos2/PosthocUQ>)
- [ ] ~~Improving the reliability for confidence estimation [[ECCV2022]](<https://arxiv.org/abs/2210.06776>)~~
- [ ] Gradient-based Uncertainty for Monocular Depth Estimation [[ECCV2022]](<https://arxiv.org/abs/2208.02005>) - [[PyTorch]](<https://github.com/jhornauer/GrUMoDepth>)
- [ ] BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks [[ECCV2022]](<https://arxiv.org/abs/2207.06873>) - [[PyTorch]](<https://github.com/ExplainableML/BayesCap>)
- [ ] ~~Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model [[AAAI2022]](<https://ojs.aaai.org/index.php/AAAI/article/view/20773>)~~
- [ ] ~~Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation [[NeurIPS2022]](<https://openreview.net/pdf?id=epjxT_ARZW5>)~~
- [ ] Learning Structured Gaussians to Approximate Deep Ensembles [[CVPR2022]](<https://arxiv.org/abs/2203.15485>)
- [ ] Learning Uncertainty For Safety-Oriented Semantic Segmentation In Autonomous Driving [[ICIP2022]](<https://arxiv.org/abs/2105.13688>)
- [ ] SLURP: Side Learning Uncertainty for Regression Problems [[BMVC2021]](<https://arxiv.org/pdf/2110.11182.pdf>) - [[PyTorch]](<https://github.com/xuanlongORZ/SLURP_uncertainty_estimate>)
- [ ] Learning to Predict Error for MRI Reconstruction [[MICCAI2021]](<https://arxiv.org/abs/2002.05582>)
- [ ] ~~Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation [[ICCV2021]](<https://arxiv.org/abs/2108.01634>)~~ - [[PyTorch]](<https://github.com/valeoai/obsnet>)
- [ ] ~~A Mathematical Analysis of Learning Loss for Active Learning in Regression [[CVPR Workshop2021]](<https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html>)~~
- [ ] Real-time uncertainty estimation in computer vision via uncertainty-aware distribution distillation [[WACV2021]](<https://arxiv.org/abs/2007.15857>)
- [ ] ~~Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel [[ICLR2020]](<https://arxiv.org/abs/1906.00588>) - [[TensorFlow]](<https://github.com/cognizant-ai-labs/rio-paper>)~~
- [ ] ~~Gradients as a Measure of Uncertainty in Neural Networks [[ICIP2020]](<https://arxiv.org/abs/2008.08030>)~~
- [ ] ~~Learning Loss for Test-Time Augmentation [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html>)~~
- [ ] ~~On the uncertainty of self-supervised monocular depth estimation [[CVPR2020]](<https://arxiv.org/abs/2005.06209>) - [[PyTorch]](<https://github.com/mattpoggi/mono-uncertainty>)~~
- [ ] DEUP: Direct Epistemic Uncertainty Prediction [[arXiv2020]](<https://arxiv.org/abs/2102.08501>)
- [ ] Addressing failure prediction by learning model confidence [[NeurIPS2019]](<https://papers.NeurIPS.cc/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf>) - [[PyTorch]](<https://github.com/valeoai/ConfidNet>)
- [ ] ~~Learning loss for active learning [[CVPR2019]](<https://arxiv.org/abs/1905.03677>) - [[PyTorch]](<https://github.com/Mephisto405/Learning-Loss-for-Active-Learning>) (unofficial codes)~~
- [ ] ~~Structured Uncertainty Prediction Networks [[CVPR2018]](<https://arxiv.org/abs/1802.07079>) - [[TensorFlow]](<https://github.com/Era-Dorta/tf_mvg>)~~
- [ ] ~~Uncertainty estimates and multi-hypotheses networks for optical flow [[ECCV2018]](<https://arxiv.org/abs/1802.07095>) - [[TensorFlow]](<https://github.com/lmb-freiburg/netdef_models>)~~
- [ ] ~~Classification uncertainty of deep neural networks based on gradient information [[IAPR Workshop2018]](<https://arxiv.org/abs/1805.08440>)~~
- [ ] ~~What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[NeurIPS2017]](<https://arxiv.org/abs/1703.04977>)~~
- [ ] Estimating the Mean and Variance of the Target Probability Distribution [[(ICNN94)]](<https://ieeexplore.ieee.org/document/374138>)


## Dirichlet-networks/Evidential-deep-learning
- [ ] ~~Uncertainty Estimation by Fisher Information-based Evidential Deep Learning [[ICML2023]](https://arxiv.org/pdf/2303.02045.pdf) - [[PyTorch]](<https://github.com/danruod/iedl>)~~
- [ ] ~~Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification [[CVPR2023]](https://arxiv.org/abs/2304.05165)~~
- [ ] (Tries to bring more theorical understanding on why EDL works)The Unreasonable Effectiveness of Deep Evidential Regression [[AAAI2023]](<https://arxiv.org/abs/2205.10060>)
- [ ] Fast Predictive Uncertainty for Classification with Bayesian Deep Networks [[UAI2022]](<https://arxiv.org/abs/2003.01227>) - [[PyTorch]](<https://github.com/mariushobbhahn/LB_for_BNNs_official>)
- [ ] ~~An Evidential Neural Network Model for Regression Based on Random Fuzzy Numbers [[BELIEF2022]](<https://arxiv.org/abs/2208.00647>)~~ (Only the introduction for Dempster)
- [ ] Natural Posterior Network: Deep Bayesian Uncertainty for Exponential Family Distributions [[ICLR2022]](<https://arxiv.org/abs/2105.04471>) - [[PyTorch]](<https://github.com/borchero/natural-posterior-network>)
- [ ] ~~Improving Evidential Deep Learning via Multi-task Learning [[AAAI2022]](<https://arxiv.org/abs/2112.09368>)~~
- [ ] ~~Trustworthy multimodal regression with mixture of normal-inverse gamma distributions [[NeurIPS2021]](<https://arxiv.org/abs/2111.08456>)~~
- [ ] ~~Misclassification Risk and Uncertainty Quantification in Deep Classifiers [[WACV2021]](<https://openaccess.thecvf.com/content/WACV2021/html/Sensoy_Misclassification_Risk_and_Uncertainty_Quantification_in_Deep_Classifiers_WACV_2021_paper.html>)~~
- [ ] Evaluating robustness of predictive uncertainty estimation: Are Dirichlet-based models reliable? [[ICML2021]](<http://proceedings.mlr.press/v139/kopetzki21a/kopetzki21a.pdf>)
- [ ] Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts  [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html>) - [[PyTorch]](<https://github.com/sharpenb/Posterior-Network>)
- [ ] ~~Being Bayesian about Categorical Probability [[ICML2020]](<http://proceedings.mlr.press/v119/joo20a/joo20a.pdf>)~~
- [ ] Ensemble Distribution Distillation [[ICLR2020]](<https://arxiv.org/abs/1905.00076>)
- Conservative Uncertainty Estimation By Fitting Prior Networks [[ICLR2020]](<https://openreview.net/forum?id=BJlahxHYDS>)
- [ ] ~~Noise Contrastive Priors for Functional Uncertainty [[UAI2020]](<https://proceedings.mlr.press/v115/hafner20a.html>)~~
- [ ] Deep Evidential Regression [[NeurIPS2020]](<https://arxiv.org/abs/1910.02600>) - [[TensorFlow]](<https://github.com/aamini/evidential-deep-learning>)
- [ ] ~~Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples [[NeurIPS Workshop2020]](<https://arxiv.org/abs/2010.10474>)~~
- [ ] ~~Uncertainty on Asynchronous Time Event Prediction [[NeurIPS2019]](<https://arxiv.org/abs/1911.05503>) - [[TensorFlow]](<https://github.com/sharpenb/Uncertainty-Event-Prediction>)~~
- [ ] Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness [[NeurIPS2019]](<https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html>)
- [ ] ~~Quantifying Classification Uncertainty using Regularized Evidential Neural Networks [[AAAI FSS2019]](<https://arxiv.org/abs/1910.06864>)~~
- [ ] ~~Evidential Deep Learning to Quantify Classification Uncertainty [[NeurIPS2018]](<https://arxiv.org/abs/1806.01768>) - [[PyTorch]](<https://github.com/dougbrion/pytorch-classification-uncertainty>)~~
- [ ] Predictive uncertainty estimation via prior networks [[NeurIPS2018]](<https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html>)
- [ ] A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation [[arXiv2021]](<https://arxiv.org/abs/2110.03051>)

## Deterministic-Uncertainty-Methods
- [ ] Deep Deterministic Uncertainty: A Simple Baseline [[CVPR2023]](<https://arxiv.org/abs/2102.11582>) - [[PyTorch]](<https://github.com/omegafragger/DDU>)
- [ ] Training, Architecture, and Prior for Deterministic Uncertainty Methods [[ICLR Workshop2023]](<https://arxiv.org/abs/2303.05796>) - [[PyTorch]](<https://github.com/orientino/dum-components>)
- [ ] Latent Discriminant deterministic Uncertainty [[ECCV2022]](<https://arxiv.org/abs/2207.10130>) - [[PyTorch]](<https://github.com/ENSTA-U2IS/LDU>)
- [ ] Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression [[CoRR2021]](<https://arxiv.org/abs/2102.11409>)
- [ ] ~~Training normalizing flows with the information bottleneck for competitive generative classification [[NeurIPS2020]](<https://arxiv.org/abs/2001.06448>)~~
- [ ] ~~Simple and principled uncertainty estimation with deterministic deep learning via distance awareness [[NeurIPS2020]](<https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html>)~~
- [ ] Uncertainty Estimation Using a Single Deep Deterministic Neural Network [[ICML2020]](<https://arxiv.org/abs/2003.02037>) - [[PyTorch]](<https://github.com/y0ast/deterministic-uncertainty-quantification>)
- [ ] ~~Single-Model Uncertainties for Deep Learning [[NeurIPS2019]](<https://arxiv.org/abs/1811.00908>) - [[PyTorch]](<https://github.com/facebookresearch/SingleModelUncertainty/>)~~
- [ ] ~~Sampling-Free Epistemic Uncertainty Estimation Using Approximated Variance Propagation [[ICCV2019]](<https://openaccess.thecvf.com/content_ICCV_2019/html/Postels_Sampling-Free_Epistemic_Uncertainty_Estimation_Using_Approximated_Variance_Propagation_ICCV_2019_paper.html>) - [[PyTorch]](<https://github.com/janisgp/Sampling-free-Epistemic-Uncertainty>)~~
