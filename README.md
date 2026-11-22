# MA289 Word Embeddings Competition:
This repository contains the complete experimentation workflow and final model implementation for the MA289 Word Embedding Competition. The beginning of the Colab notebook include main ideas of how to do neural networking. Following the main ideas provides by our instructor, the following was experiments in different directions in attempt to optimize our baseline model. Our optimized baseline, Joint Opt B, achieved our top competitive performance on the leaderboard:

Private Score (MSE): 0.06968

Public Score (MSE): 0.07106

📖 Project Overview

The goal of the competition was to learn high-quality word representations (embeddings) from a text corpus containing anonymous tokens, and predict token-pair similarities evaluated using Mean Squared Error (MSE).


The pipeline transitions through:

Anonymous Token⟶Integer ID⟶nn.Embedding⟶Context Prediction⟶Relative Geometry Similarity


🛠️ Key Architectural Insights

Shallow vs. Deep Architectures: While deep multi-layer neural networks work well for large-scale datasets, they overfit the sparse co-occurrence matrix on this smaller vocabulary (900 words). A shallow, elegant linear projection network ( E∈RV×d ) acted as a robust matrix factorization method, capturing clear semantic representations.


Angle vs. Magnitude (Cosine vs. Raw Dot Product):


Cosine Similarity isolates purely angular differences, ignoring vector lengths.

Raw Dot Products preserve vector magnitudes. Frequently updated words naturally grow longer, implicitly encoding occurrence frequencies and semantic density into the representation geometry.

The Winning Recipe (Joint Opt B Configuration): Our highest-performing approach utilized a standard 50-dimensional embedding space regularized with tight weight decay ( 10−4 ) to stabilize coordinate growth, combined with an exponential power scaling ( x1.5 ) over the mapped similarities to filter out background noise.
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
EPOCHS = 15
