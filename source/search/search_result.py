class SearchResult:
    def __init__(self, score, t5_rerank_score, sbert_rerank_score, context, sentence, news_id, mention_to_question_weight,
                 mention_to_answer_weight, turn_to_question_weight, turn_to_answer_weight,
                 mention_choice, turn_choice, answer_mentions, answer_mention_types):
        self.score = score
        self.t5_rerank_score = t5_rerank_score
        self.sbert_rerank_score = sbert_rerank_score
        self.context = context
        self.sentence = sentence
        self.news_id = news_id
        self.mention_to_question_weight = mention_to_question_weight
        self.mention_to_answer_weight = mention_to_answer_weight
        self.turn_to_question_weight = turn_to_question_weight
        self.turn_to_answer_weight = turn_to_answer_weight
        self.mention_choice = mention_choice
        self.turn_choice = turn_choice
        self.answer_mentions = answer_mentions
        self.answer_mention_types = answer_mention_types
