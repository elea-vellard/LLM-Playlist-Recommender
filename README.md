# Language Model-Based Playlist Generation Recommender

This repository contains the implementation of a novel approach to **playlist generation using language models**. Our method leverages the **thematic coherence between playlist titles** and their tracks by creating **semantic clusters** from text embeddings. We **fine-tune a transformer model** on these clusters to generate playlists based on cosine similarity scores between known and unknown titles, utilizing a **voting mechanism** for final recommendations.

The repository includes code for preprocessing, model training, evaluation, and generating recommendations. For more detail, please refer to the [paper](#citation).

**Related links**
- [Online demo](https://playlist-recommendation.tools.eurecom.fr/) - [code](https://github.com/elea-vellard/DEMO-playlist-continuation)
- [Zenodo repository](https://zenodo.org/records/15837980) including the best trained model.

## 1. Transform and pre-process the dataset

Run:

```bash
python3 transform-dataset/json2csv.py
```

to convert the JSON slices of the dataset into user-friendly CSV files:

## 2. Embedding generation and clustering

First, playlists titles and tracks are embedded using a pre-trained SentenceBERT model and stored in a 'pickle' file:

```bash
python3 clustering-no-split/embeddings/track_embeddings_no-split.py
```

Then, the K-means clustering algorithm is applied to create the clusters, and the generated 'csv' file is modified to calculate and include the percentage of exact matches:

```bash
python3 clustering-no-split/clusters/clustering-no-split.py clustering-no-split/clusters/percent-no-split.py
```

Apply the clean algorithm to remove miscellaneous clusters:

```bash
python3 clustering-no-split/clean/clean.py
```

Finally, randomly split the clusters, ensuring a representation of each cluster in both train, test and validation sets:

```bash
python3 clustering-no-split/split/split.py
```

## 3. Finetuning

Train the SentenceBERT model with two loss functions (cross-entropy and triplet loss) to better capture thematic similarities:

```bash
python3 finetuning/cross_entropy_model_finetuning.py finetuning/finetuning_triplet_loss.py
```

## 4. Generate the embeddings for playlists titles using the fine-tuned models

Run :

```bash
python3 embeddings/playlists_embeddings_final.py
```

to generate embeddings for playlist titles using the fine-tuned models.
> Make sure to adjust the model path to select either the triplet loss model, the cross-entropy loss model or the pretrained model.

## 5. Generate the recommendations and evaluate the models

Evaluate the metrics for a given test playlist:

```bash
python3 similarity/test_1_playlist_finetuned_model.py
```

Generate the recommendation for a playlist title:

```bash
python3 similarity/recommend.py
```

Assess the model’s overall performance on the complete test set:

```bash
python3 similarity/testset_test_model.py
```

> Make sure to adjust the model path to select either the triplet loss model, the cross-entropy loss model or the pretrained model.

## Citation

If you use this software, please cite ([bib file](https://raw.githubusercontent.com/elea-vellard/LM-Playlist-Recommender/refs/heads/main/charoloisvellard2025llm-recommender.bib)):

Enzo Charolois–Pasqua, Eléa Vellard, Youssra Rebboud, Pasquale Lisena,
and Raphael Troncy. 2025. **A Language Model-Based Playlist Generation
Recommender System**. In *Proceedings of the Nineteenth ACM Conference on
Recommender Systems (RecSys ’25)*, September 22–26, 2025, Prague, Czech Re-
public. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3705328.3748053
