# PrtrainedBA

PretrainedBA is a complex-free binding affinity prediction model. \
To improve the accuracy of binding affinity prediction, each model that makes up PretrainedBA is pretrained with data from external datasets.
Due to capacity issues, train and test data is available for download from the following sources: 

Replace the 'data' dir within the GitHub folder with the downloaded 'data' dir to utilize the provided code for train and test.


![KakaoTalk_20240716_222027048](https://github.com/user-attachments/assets/7d6225a3-c801-444f-a43e-d72fd0f3d3a2)


## Requirements

python==3.7 \
Pytorch==1.7.1 \
RDKit==2021.03.01 \
Openbabel==2.4.1 \
[ProtTrans](https://github.com/agemagician/ProtTrans)


## Binding affinity prediction using trained model

The weights of PretrainedBA trained on the 5-fold cross-validation dataset are provided in the 'checkpoints/affinity' directory. The trained model can be used to predict binding affinity for the given compound-protein interactions. 
We provide code to predict binding affinity for eight test datasets, including the internal test dataset (see __10.BA_test.py__).

The prediction results for the eight test datasets using PretrainedBA can be found in the following path: 'results/affinity' 

The ability to accurately predict binding affinity and interpret interaction is an important evaluation of complex-free models. We evaluated these capabilities on two test datasets using PretrainedBA (see __11.Interactionsite_test.py__). 


## Training PretrainedBA
PretrainedBA undergoes a three-stage pre-training process. After pre-training is completed in each stage, the models from those stages are combined and trained to predict binding affinity.
More details can be found in the original paper. 

We provide the code for training PretrainedBA in seven steps as shown below. Steps marked as optional are for pre-training.
Also, we provide pre-trained models for each stage in the 'checkpoints' dir, which you can skip the pre-training stages if you only want to train binding affinity. 

### 1. Get protein amino acid-level embeddings
PretrainedBA uses amino acid-level embeddings extracted from protein sequences.
The amino acid-level embeddings are extracted using the ProtTrans model, which is pre-trained on a large-scale sequence database.
The code for extracting embeddings uses __01.Get_protein_features.py__.


### 2. Get compound atom-level features 
PretrainedBA uses molecular graphs to predict binding affinity. The code to extract atom-level features from the given molecular graphs uses __02.Get_compound_features.py__. 


### 3. (optional) Compound encoder pre-training
There are two main steps to the compound encoder training. First, we leverage VQVAE to tokenize compound atoms into discrete codes in a contex-aware manner. The compound encoder is then trained using masked atom modeling (MAM) and triplet masked contrastive learning (TMCL).

The compound encoder is trained using about 2000K compounds selected from the Zinc database. The code for training uses __03.Compound_VQVAE_training.py__ and __04.Compound_encoder_pretraining.py__. 
The trained compound encoder is provided by 'results/pre-training/compound' dir.


### 4. (optional) Pocket predictor pre-training
To predict binding pockets from protein sequences, we utilize Pseq2Sites, a state-of-the-art prediction model, as a pocket predictor. A more detailed description of Pseq2Sites can be found in the original paper.  

Pocket predictor is trained using data from about 80,000 compound-protein complex structures curated from the PDB. The code for training uses __05.Pocket_predictor_pretraining.py__.

The threshold setting, which is a hyperparameter of the trained pocket predictor, and the binding pocket extraction use the __06.Pocket_test.py__ and __07.Get_pocket_info.py__ codes. 


### 5. (optional) Interaction site predictor (also protein encoder and cross-attention encoder) pre-training
The interaction site predictor is trained using binding pocket and pre-trained compound encoder. The binding pocket is extracted by the pre-trained pocket predictor. The training data consists of about 80,000 compound-protein complex structures curated from the PDB.

Interaction site labels are extracted using the protein-ligand interaction profiler, PLIP. The code for training uses __08.Interactionsite_predictor_pretraining.py__. 


### 6. Training on binding affinty dataset 
Use each of the pre-trained modules to train PretrainedBA, a model for binding affinity prediction. The code for training uses __09.BA_training.py__. 


