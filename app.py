from flask import Flask
from flask_cors import CORS
from graphql_server.flask import GraphQLView

from pathlib import Path

from data_handler import load_user_artists, ArtistRetriever
from recommender import ALSRecommender
from schema import schema

app = Flask(__name__)
CORS(app, resources={r"/graphql/*": {"origins": "*"}})

# Global instance of ALSRecommender to be used across requests
user_artists = load_user_artists(Path("./data/user_artists.dat"))
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("./data/artists.dat"))

als_recommender = ALSRecommender(artist_retriever)
als_recommender.fit(user_artists)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# # GraphQL endpoint
app.add_url_rule(
    '/graphql', 
    view_func=GraphQLView.as_view(
        'graphql', 
        schema=schema, 
        graphiql=True
    )
)
