"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from data_handler import load_user_artists, ArtistRetriever


class ImplicitRecommender:
    """
    This class computes recommendations for a given user
    using the implicit library to provide recommendations for artists using collaborative filtering
    See: https://benfred.github.io/implicit/api/models/recommender_base.html

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
            - we use the base class to let the client code choose which model they want to use during initialiazation
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        # See https://benfred.github.io/implicit/api/models/recommender_base.html
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """
        Takes in a user and the data matrix. Then returns a list of artists and a list of scores (ratings) for the artist
        """
        # Return the top n recommendations for the given user.
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]

        return artists, scores

class ALSRecommender(ImplicitRecommender):
    pass
    """
    This class extends ImplicitRecommender to specifically use the Alternating Least Squares (ALS) model from the implicit library
    for artist recommendations. It adds a method to list 50 recommended artists given an artist ID.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an instance of implicit.als.AlternatingLeastSquares
    """

    def __init__(self, artist_retriever: ArtistRetriever):
        als_model = implicit.als.AlternatingLeastSquares(
            factors=50, iterations=10, regularization=0.01
        )
        super().__init__(artist_retriever, als_model)
        
    def list_50_artists(self, user_id: int, user_artists_matrix: scipy.sparse.csr_matrix) -> Tuple[List[str], List[float]]:
        """
        Recommends 50 artists for a given artist ID by utilizing the recommend method inherited from the parent class.

        Parameters:
        - user_id: The ID of the user to get recommendations for.
        - user_artists_matrix: The user-artist interaction matrix.

        Returns:
        - A tuple of two lists: one for artist names and one for their corresponding recommendation scores.
        """
        return self.recommend(user_id, user_artists_matrix, n=50)

if __name__ == "__main__":

    # load user artists matrix
    user_artists = load_user_artists(Path("./data/user_artists.dat"))

    # create artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./data/artists.dat"))

    # create alternating least squares model from implicit
    # factors: the number of latent factors to use
    # iterations: the number of iterations to use when fitting data (each iteration is 1 step of fix A and optimize B, then fix B and optimize A)
    # regularization: a factor to prevent overfitting or poor performance
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    ALS_recommender = ALSRecommender(artist_retriever)
    ALS_recommender.fit(user_artists)
    
    recommendations = ALS_recommender.list_50_artists(1, user_artists)
    print(recommendations)