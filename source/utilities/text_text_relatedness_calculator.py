from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

text_to_text_model = SentenceTransformer('all-mpnet-base-v2')
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=128)


def get_text_to_text_related_score(a_text, text_list):
    a_text_embedding = text_to_text_model.encode(a_text)
    text_list_embedding = text_to_text_model.encode(text_list)
    return util.cos_sim(a_text_embedding, text_list_embedding).numpy().flatten().tolist()


def get_question_to_answer_related_score(pairs):
    scores = cross_encoder.predict(pairs)
    return scores.flatten().tolist()


if __name__ == "__main__":
    print(get_text_to_text_related_score("Thuringen HC is team, sportsteam, organization.", [
        "Rooney scored more than 1000 goals for Thuringen HC.",
        "How many goals scored for the team?"]
    ))
