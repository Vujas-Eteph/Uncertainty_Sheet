### 2023
 - **Packed-Ensembles(ICLR 2023)**
   - Make Ensemble more memory and inference efficient
   -  

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


### 2020
 - **Batch Ensemble (ICLR 2020)**:
   - Tackle the bottleneck of memory consumption and inference speed. 
   - Leads to speed up and less memory consumption compared to Deep Ensemble.
   - Instead of using multiple models, use one model (i.e., the weights $W$) to generate an ensemble of models. To generate the new weights of the model, multiply the weights with a generated mask $F_{i} = s_{i}r_{i}^T$ so that the new weights are $W_{i} = W \circ F_{i}$. Here, $s_{i}$,$r_{i}$ are learnable parameters during training.
   - Hence they also don't need to store the masks $F_{i:N}$ (Similar to MC Dropout, in my opinion) but only the vectors that generate them, which takes less memory.
   - The efficiency comes from the matrix multiplication/vectorization, where all the models can be passed in parallel for a prediction, and they all use the same "backbone model" to generate the ensemble methods.
  
