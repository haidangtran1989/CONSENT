from eval.judgement_loader import get_judgements
import pytrec_eval
import os
import collections
import sys

CUT_AT = int(sys.argv[1])

run = dict()


def extract_scores(conversation_turn):
    pos = conversation_turn.find("\n\nTo be continued...\n\n")
    if pos >= 0:
        conversation_turn = conversation_turn[:pos]
        conversation_turn = conversation_turn.strip()
    if conversation_turn == "":
        return
    lines = conversation_turn.split("\n")
    for k in range(2, min(len(lines), 2 + CUT_AT)):
        parts = lines[k].strip().split("\t")
        score = float(parts[0])
        q_id = parts[-1]
        sent_id = parts[-2]
        actual_score = run.get(q_id, dict())
        actual_score[sent_id] = score
        run[q_id] = actual_score


folder_path = "../data/result/consent_test/"
file_names = os.listdir(folder_path)
file_names.sort()
for file_name in file_names:
    with open(folder_path + file_name, "r", encoding="utf-8") as judge_file:
        content = judge_file.read()
        content = content.strip()
        conversation_turns = content.split("+++++")
        for conversation_turn in conversation_turns:
            extract_scores(conversation_turn.strip())


def print_effectiveness(title, result, metric, name):
    result = collections.OrderedDict(sorted(result.items()))
    with open("../data/eval/" + name + ".txt", "wt", encoding="utf-8") as output_file:
        output_file.write(title + "\n")
        sum_effectiveness = 0.0
        count = 0
        for query_id, effectiveness in result.items():
            output_file.write(query_id + "\t" + str(effectiveness[metric]) + "\n")
            sum_effectiveness += effectiveness[metric]
            count += 1
        output_file.write(str(count) + "\n")
        output_file.write(str(sum_effectiveness / count) + "\n")
        output_file.write("+++++\n")


query_id_to_judgements = get_judgements()
mrr_evaluator = pytrec_eval.RelevanceEvaluator(query_id_to_judgements, {"recip_rank"}, relevance_level=1)
print_effectiveness("MRR measure", mrr_evaluator.evaluate(run), "recip_rank", "sbert_zero_mrr_" + str(CUT_AT))

if CUT_AT > 1:
    ndcg_evaluator = pytrec_eval.RelevanceEvaluator(query_id_to_judgements, {"ndcg_cut.3"}, relevance_level=1)
    print_effectiveness("nDCG measure", ndcg_evaluator.evaluate(run), "ndcg_cut_3", "sbert_zero_ndcg_" + str(CUT_AT))
