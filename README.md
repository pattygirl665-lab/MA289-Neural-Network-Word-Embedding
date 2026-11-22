# MA289 Word Embeddings Competition: Repo README
This repository contains the complete experimentation workflow and final model implementation for the MA289 Word Embedding Competition. Our optimized baseline, Joint Opt B, achieved our top competitive performance on the leaderboard:

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

📊 Summary of Selected Key Experiments
Configuration / Model	Dimensions	Normalization & Scaling Method	Regularization	Performance Status
Baseline	50	Cosine  →  Linear Shift (cos_sim + 1.0) / 2.0	None	Initial benchmark
Deeper Network	64	Cosine  →  Linear Shift	Overfit (35 epochs)	High validation loss
Optimized Baseline	50	Cosine  →  Dynamic Min-Max	Weight Decay  1e−5 	Improved distribution fit
Joint Opt A	32	Cosine  →  Dynamic Min-Max	Weight Decay  1e−5 	Strong generalizability
Joint Opt B	50	Cosine  →  Power Scaling ( 1.5 )	Weight Decay  1e−4 	Best Submission (Private: 0.06968)
🚀 Getting Started
Prerequisites
Make sure your environment has PyTorch, Pandas, NumPy, and Scikit-Learn installed:

pip install torch pandas numpy scikit-learn
Running the Training Script
To reproduce the best-performing Joint Opt B submission, run the respective PyTorch training cell containing your target hyperparameters:

EMBEDDING_DIM = 50
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
EPOCHS = 15
