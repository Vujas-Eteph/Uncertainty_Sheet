
### 2021
 - **Masksembles (CVPR 2021)** [Video](https://www.youtube.com/watch?v=YWKVdn3kLp0):
   - Another name would be more suitable, more related to MC dropout ??
   - Want to tackle uncertainty estimation with Deep Ensemble, but with lower inferences and training times! Hence, reducing the computational cost.
   - Introduce a structured approach for dropping model parameters (versus MC Dropout, which is random).
   - Acts as a continuum between single models, deep-ensemble and MC dropout.
   - 3 parameters to look at when generating masks:
     - N: Number of masks
     - M: Number of ones in each mask
     - S: Amount of overlap between the masks (Similar to IoU between the generated masks and applying a threshold ? Seems to me in 3.4.)
   - Important: Don't use any number for N,M,S as this implies that the network has to be adapted each time you change N,M,S. Instead, please make sure you have SxM equal to the original amount of channels as in the original NN.
   - Complete overlap leads to a single model.
   - Allowing a fair amount of overlap between the model's weights leads to MC dropout.
   - Avoiding mask overlaps leads to a Deep Ensemble method.
