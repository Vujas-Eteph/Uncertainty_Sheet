## Conglomerate of notes and ideas collected during my research in the field. 
*(If u notice an error (i.e., false explanation, statement, reference or whatever) please report it to me.)*

### General Notes on Deep Ensemble
- Deep Ensemble produces more reliable uncertainty estimations than MC Dropout [ref. Masksembles, ref. BatchEnsemble]. Also, experiments have shown that simple ensembles have a low correlation with each other [ref. Masksembles], which is great, since we get diverse predictions for input with simple changes.
- MC Dropout is mathematically more sound than Deep Ensemble?
- Types of Uncertainties:
  - Aleatoric (data) uncertainty (is inherent to the sensor (e.g., image resolution, ...)).
  - Epistemic (model) uncertainty (can be reduced with mode training data).
  - Distributional (data shift) uncertainty (the training data only captures a portion of the real-world distribution)
- Use entropy or variance as an uncertainty estimator. (How does that work with the entropy ?)
- A good ensemble is one where the members are accurate and produce independent errors [ref. BatchEnsemble].
- DNNs trained with different initializations and SGD-like algos (although being the same model) + i.i.d images in Batches lead the models to converge towards different local minima (Due to the stochastic process and the high dimensionality of the optimization space).
- A postulate for Deep Ensembles to perform better than other methods in uncertainty quantification because they explore different modes, whereas other methods (i.e., Varional methods) explore the uncertainty only locally to one mode [ref. BatchEnsemble].
- An essential property of DE to improve predictive uncertainty estimation is related to the diversity of its predictions. 


### Quantify the quality of the uncertainty:
- Expected Calibration Error (ECE) [ref. Masksembles].

# Make a Table with positive and negative elements of each uncertainty method (Take it from a survey paper)
