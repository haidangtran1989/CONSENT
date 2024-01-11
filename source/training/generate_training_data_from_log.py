import ast
import random
from utilities.config import *
import os


def extract_features(prev_dialog_mentions, dependency_turns, dependency_mentions,
                     turn_to_question_rel, mention_to_question_rel,
                     question_to_answer_base_score, mention_to_answer_rel, turn_to_answer_rel):
    turn_to_question_sum = 0.0
    mention_to_question_sum = 0.0
    turn_to_answer_sum = 0.0
    mention_to_answer_sum = 0.0
    choice_sum = len(dependency_turns) + len(dependency_mentions)
    dependency_turn_set = set(dependency_turns)
    dependency_mention_set = set(dependency_mentions)
    turn_size = len(prev_dialog_mentions)
    k = 0
    ok = False
    for i in range(turn_size):
        if i in dependency_turn_set:
            turn_to_question_sum += turn_to_question_rel[i]
            turn_to_answer_sum += turn_to_answer_rel[i]
        for j in range(len(prev_dialog_mentions[i])):
            if i in dependency_turn_set:
                if prev_dialog_mentions[i][j] in dependency_mention_set:
                    mention_to_question_sum += mention_to_question_rel[k]
                    mention_to_answer_sum += mention_to_answer_rel[k]
                    ok = True
            k += 1
    if (not ok) and (len(dependency_turns) > 0):
        return None
    return [turn_to_question_sum, turn_to_answer_sum, mention_to_question_sum, mention_to_answer_sum, question_to_answer_base_score, -1.0 * len(dependency_turns), -1.0 * len(dependency_mentions)]


def generate_noise(prev_dialog_mentions, precise_dependency_mentions):
    dependency_turns = set()
    dependency_mentions = set()
    if MAX_RELEVANT_TURNS_OR_ENTITIES is not None:
        size_cap = random.randint(0, MAX_RELEVANT_TURNS_OR_ENTITIES * 2)
        size_cap = min(size_cap, MAX_RELEVANT_TURNS_OR_ENTITIES)
    else:
        total_entity_num = sum([len(x) for x in prev_dialog_mentions])
        size_cap = random.randint(0, total_entity_num)
    turn_size = len(prev_dialog_mentions)
    if random.randint(0, 2) == 2:
        # Sample on different turns
        for k in range(size_cap):
            i = random.randint(0, turn_size - 1)
            if len(prev_dialog_mentions[i]) > 0:
                j = random.randint(0, len(prev_dialog_mentions[i]) - 1)
                if prev_dialog_mentions[i][j] not in precise_dependency_mentions:
                    dependency_turns.add(i)
                    dependency_mentions.add(prev_dialog_mentions[i][j])
                else:
                    return None, None, None
    else:
        # Sample on the same turns
        i = random.randint(0, turn_size - 1)
        for k in range(size_cap):
            j = random.randint(0, len(prev_dialog_mentions[i]) - 1)
            if prev_dialog_mentions[i][j] not in precise_dependency_mentions:
                dependency_turns.add(i)
                dependency_mentions.add(prev_dialog_mentions[i][j])
            else:
                return None, None, None
    return dependency_turns, dependency_mentions, size_cap


def add_and_filter(mentions, new_mention):
    for i in range(len(mentions)):
        if new_mention in mentions[i]:
            return
    mentions.append(new_mention)


def parse_search_log(file_name):
    with open(file_name, "rt", encoding="utf-8") as log_file:
        content = log_file.read().strip()
    rank_results = content.split("+++++")
    for i in range(0, len(rank_results)):
        rank_result = rank_results[i].strip()
        if "To be continued..." not in rank_result:
            continue
        rank_result = rank_result[rank_result.find("To be continued...") + len("To be continued..."):].strip()
        lines = rank_result.strip().split("\n")
        dependency_mention = lines[2][len("Dependency Entity: "):]
        prev_dialog_mentions = ast.literal_eval(lines[3][len("Prev Dialog Mentions: "):])
        j = 16
        relevant_stats_list = list()
        irrelevant_stats_list = list()
        turn_to_question_rel = None
        mention_to_question_rel = None
        golden_answer = None
        while j < len(lines):
            parts = lines[j].split("\t")
            if len(parts) == 1:
                break
            question_to_answer_base_score = float(parts[2])
            mention_to_question_rel = ast.literal_eval(lines[j + 1][len("Mention to question weight: "):])
            mention_to_answer_rel = ast.literal_eval(lines[j + 2][len("Mention to answer weight: "):])
            turn_to_question_rel = ast.literal_eval(lines[j + 3][len("Turn to question weight: "):])
            turn_to_answer_rel = ast.literal_eval(lines[j + 4][len("Turn to answer weight: "):])
            stats = (question_to_answer_base_score, mention_to_answer_rel, turn_to_answer_rel)
            if parts[5] == "N/A" or parts[5].startswith("A_"):
                relevant_stats_list.append(stats)
                golden_answer = parts[3]
            else:
                irrelevant_stats_list.append(stats)
            j += 10
        irrelevant_stats_list.sort(key=lambda x: x[0], reverse=True)
        irrelevant_stats_list = irrelevant_stats_list[3:]
        if golden_answer is None:
            continue
        dependency_mentions = [dependency_mention]
        for k in range(len(prev_dialog_mentions)):
            for l in range(len(prev_dialog_mentions[k])):
                if prev_dialog_mentions[k][l] in golden_answer:
                    add_and_filter(dependency_mentions, prev_dialog_mentions[k][l])
        dependency_turns = set()
        for mention in dependency_mentions:
            max_turn_rel = 0
            max_k = -1
            for k in range(len(prev_dialog_mentions)):
                if mention in prev_dialog_mentions[k]:
                    if max_turn_rel < turn_to_question_rel[k]:
                        max_turn_rel = turn_to_question_rel[k]
                        max_k = k
            if max_k >= 0:
                dependency_turns.add(max_k)
        if len(dependency_turns) == 0:
            continue
        dependency_turns = list(dependency_turns)
        if MAX_RELEVANT_TURNS_OR_ENTITIES is not None and len(dependency_mentions) > MAX_RELEVANT_TURNS_OR_ENTITIES:
            continue
        for relevant_stats in relevant_stats_list:
            pos_features = extract_features(prev_dialog_mentions, dependency_turns, dependency_mentions,
                                            turn_to_question_rel, mention_to_question_rel,
                                            relevant_stats[0], relevant_stats[1], relevant_stats[2])
            if pos_features is None:
                continue
            if (MAX_RELEVANT_TURNS_OR_ENTITIES is None) or (MAX_RELEVANT_TURNS_OR_ENTITIES is not None and len(dependency_mentions) <= MAX_RELEVANT_TURNS_OR_ENTITIES):
                for k in range(200):
                    new_dependency_turns, new_dependency_mentions, size_cap = generate_noise(prev_dialog_mentions, dependency_mentions)
                    if new_dependency_mentions is None:
                        continue
                    pos_features_with_noises = extract_features(prev_dialog_mentions, new_dependency_turns, new_dependency_mentions,
                                                                turn_to_question_rel, mention_to_question_rel,
                                                                relevant_stats[0], relevant_stats[1], relevant_stats[2])
                    if pos_features_with_noises is None:
                        continue
                    if size_cap > 0 and pos_features_with_noises[2] / size_cap > pos_features[2] / len(dependency_mentions) + 0.05:
                        continue
                    print(pos_features)
                    print(pos_features_with_noises)
                    print("vs adding noise")
                    print("-----")
            for irrelevant_stats in irrelevant_stats_list:
                new_dependency_turns, new_dependency_mentions, size_cap = generate_noise(prev_dialog_mentions, dependency_mentions)
                if new_dependency_mentions is None or random.randint(0, 3) == 2:
                    new_dependency_mentions = dependency_mentions
                    new_dependency_turns = dependency_turns
                    size_cap = len(dependency_mentions)
                neg_features = extract_features(prev_dialog_mentions, new_dependency_turns, new_dependency_mentions,
                                                turn_to_question_rel, mention_to_question_rel,
                                                irrelevant_stats[0], irrelevant_stats[1], irrelevant_stats[2])
                if neg_features is None:
                    continue
                if size_cap > 0 and neg_features[2] / size_cap > pos_features[2] / len(dependency_mentions) + 0.05:
                    continue
                if neg_features[4] < MIN_BASE_SCORE:
                    continue
                print(pos_features)
                print(neg_features)
                print("vs negative example")
                print("-----")


def parse_search_log_from_folder(folder_path):
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = folder_path + file_name
        parse_search_log(file_path)


if __name__ == "__main__":
    # python -m training.generate_training_data_from_log >../data/conv_feature_vec_train.txt
    parse_search_log_from_folder("../data/result/consent_train/")
