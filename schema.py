from graphene import ObjectType, String, Float, Schema, List, Int
import app

class RecommendationType(ObjectType):
    artist_name = String(required=True)
    score = Float(required=True)

# Define the main query class
class Query(ObjectType):
    # Query to get all artists
    get_all_artists = List(String)

    # Query to get recommendations based on an artist ID
    get_recommendations = List(RecommendationType, user_id=Int(required=True))

    def resolve_get_all_artists(self, info):
        return app.artist_retriever.list_artists()

    def resolve_get_recommendations(self, info, user_id: int):
        artists, scores = app.als_recommender.list_50_artists(user_id, app.user_artists)
        result = [{"artist_name": artist, "score": score} for artist, score in zip(artists, scores)]
        return result

# Instantiate the schema
schema = Schema(query=Query)