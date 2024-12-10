# LLM-project
Repo for the LLM-based playlist recommender system

Steps:

1. Pre-processing of data: Run transform-dataset/json2csv.py to transform initial json slices into csv files (more interpretable), then use transform-dataset/data-split.py to split the dataset into 3 sets (train, validation, test)

2. clustering: Run clustering/embeddings/tracks_embeddings.py to compute the embeddings of every tracks and store them in picle files (for faster computation of clusters) using pre-trained sentence bert.
   Run clustering/clusters/clustering.py to compute 3 files of clusters (.csv files) for the sets of train, validation and test.

3. finetuning: Run finetuning/finetuning.py to finetune the sentence bert model.
   Run finetuning/plot-loss/plot-loss.py to plot the loss of the trained model (train and validation losses).

4. embeddings: Run embeddings/playlists_embeddings.py to compute the embeddings of each playlist's title using the trained model, and store them in a pickle file.

5. Predict relevant tracks: Run similarity/test.py to generate the most relevant songs according to an input playlist name.
