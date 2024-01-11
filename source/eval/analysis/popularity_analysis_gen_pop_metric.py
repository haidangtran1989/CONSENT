import sys
import gzip
from utilities.config import MENTION_ENTITY_FILE_PATH
from utilities.nlp_utils import split_id

entity_popularity = dict()
with gzip.open(MENTION_ENTITY_FILE_PATH, "rt", encoding="utf-8") as mention_entity_file:
    for line in mention_entity_file:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        freq = int(parts[2])
        cur_total_freq = entity_popularity.get(parts[1], 0) + freq
        entity_popularity[parts[1]] = cur_total_freq


file_name = sys.argv[1]
q_id_to_metric = dict()
with open(file_name, "rt", encoding="utf-8") as file:
    content = file.read().strip()
    lines = content.split("\n")[1:-3]
    for line in lines:
        parts = line.split("\t")
        metric = float(parts[1])
        q_id = parts[0]
        q_id_to_metric[q_id] = metric


with open("../data/conv_test_set.txt", "rt", encoding="utf-8") as file:
    content = file.read().strip()
    conversations = content.split("+++++")
    for conversation in conversations:
        turns = conversation.strip().split("\n\n")
        for i in range(1, len(turns)):
            turn = turns[i].strip()
            lines = turn.split("\n")
            q_id, question = split_id(lines[0])
            _, entity = split_id(lines[3])
            popularity = entity_popularity.get(entity, -1)
            print(q_id, entity, popularity, q_id_to_metric[q_id])
