# Cross-City Latent Space Alignment for Consistency Region Embedding

These are the source codes for the **CoRE** model and its corresponding data. 

- Data
  
  raw_data/ — Data of three cities, Chengdu (CD), Beijing (BJ), and Xi'an (XA).  The data of each city contains region data, mobility data, and the socioeconomic indicator data.
  
- Code
  1. train.py — A file to run the **CoRE** model. Note that there are some hyperparameters that can be set in this file.
  2. evaluator/ — The files in this folder contain the details of cross-city region-level socioeconomic prediction tasks.
  3. model/ — These are the codes for **CoRE**, including the implementation details of all components of the model.
  
- Region Embeddings

  save_emb/ — These are the aligned embeddings of urban regions.  Running the eval.py can evaluate the alignment performance using the embeddings of different cities.
