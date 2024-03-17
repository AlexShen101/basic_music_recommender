from pathlib import Path

import scipy
import pandas as pd


def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """
    Load the user artists file and return a user-artists matrix in csr
    format.
    """

    user_artists = pd.read_csv(user_artists_file, sep="\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.weight.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()


class ArtistRetriever:
    """
    The ArtistRetriever class helps retrieve the artist name from the artist id.
    """

    def __init__(self):
        self._artists_df = None

    def load_artists(self, artists_file: Path) -> None:
        """
        Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """
        Return the artist name from the artist ID.
        """
        return self._artists_df.loc[artist_id, "name"]
    
    def get_id_from_artist_name(self, artist_name: str) -> int:
        """
        Return the artist ID from the artist name.
        """
        # Filter the DataFrame for the artist_name, and get the index (id)
        artist_ids = self._artists_df[self._artists_df['name'] == artist_name].index
        if not artist_ids.empty:
            return artist_ids[0]  # Return the first matching artist ID
        else:
            raise ValueError(f"Artist name '{artist_name}' not found.")

    def list_artists(self) -> list:
        """
        Return a list of all artist names.
        """
        return self._artists_df['name'].tolist()
