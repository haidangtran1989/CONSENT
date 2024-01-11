def get_judgements():
    judgement_file = open("../data/judges/auto_jud.consent-test.3-top.tsv", "rt", encoding="utf-8")
    query_id_to_judgements = dict()
    for line in judgement_file:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            continue
        query_id = parts[2]
        sentence_id = parts[3]
        relevance = 0
        if parts[0] == "1" and parts[1] == "1":
            relevance = 1
        query_sentence_ids_to_judgements = dict()
        if query_id in query_id_to_judgements:
            query_sentence_ids_to_judgements = query_id_to_judgements[query_id]
        query_sentence_ids_to_judgements[sentence_id] = relevance
        query_id_to_judgements[query_id] = query_sentence_ids_to_judgements
    judgement_file.close()
    return query_id_to_judgements


if __name__ == "__main__":
    print(get_judgements())
