### A conglomerate of notes and ideas collected during my research in the field. 
*(If u notice an error (i.e., false explanation, statement, reference or whatever) please report it to me.)*

- Deep Ensemble produces more reliable uncertainty estimations than MC Dropout [ref. Masksembles]. Also, experiments have shown that simple ensembles have a low correlation with each other [ref. Masksembles], which is great, since we get diverse predictions for input with simple changes.
- MC Dropout is mathematically more sound than Deep Ensemble?
- Types of Uncertainties:
  - Aleatoric (data) uncertainty (is inherent to the sensor (e.g., image resolution, ...)).
  - Epistemic (model) uncertainty (can be reduced with mode training data).
  - Distributional (data shift) uncertainty (the training data only captures a portion of the real-world distribution)
- Use entropy or variance as an uncertainty estimator. (How does that work with the entropy ?)


### Quantify the quality of the uncertainty:
- Expected Calibration Error (ECE) [ref. Masksembles].

# Make a Table with positive and negative elements of each uncertainty method (Take it from a survey paper)
