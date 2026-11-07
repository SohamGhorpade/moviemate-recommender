# Credits and resources

**Starting point.** The public *item-based-collaborative-filtering* notebook (`itcf.ipynb`) by vishwajeet-hogale
(https://github.com/vishwajeet-hogale/item-based-collaborative-filtering). I read that notebook to understand its
approach: it builds a user-by-movie matrix with missing ratings filled as 0, computes Pearson correlations between
movies, and recommends by summing similarity times the user's rating. It has no evaluation. The code in this
repository was written separately; no source files were copied from it. See `moviemate/models.py` for the list of
changes (adjusted cosine without zero-filling, shrinkage, top-k neighbors, genre and popularity models, hybrid,
chat interface, offline evaluation).

**Algorithms and ideas**
- Sarwar, Karypis, Konstan, & Riedl (2001). Item-based collaborative filtering recommendation algorithms. *WWW '01*.
- Burke (2002). Hybrid recommender systems: Survey and experiments. *User Modeling and User-Adapted Interaction, 12*(4).
- Herlocker, Konstan, Terveen, & Riedl (2004). Evaluating collaborative filtering recommender systems. *ACM TOIS, 22*(1).

**Data.** MovieLens "ml-latest-small" from GroupLens (University of Minnesota):
Harper & Konstan (2015). The MovieLens datasets: History and context. *ACM TiiS, 5*(4).
The dataset license does not allow redistribution, so the data is **not** included in this repository;
the program downloads it from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip.

**Tools.** Python, NumPy, pandas, SciPy, pytest. An AI assistant (Claude) helped draft and test code and
explain concepts; I reviewed and ran the results.
