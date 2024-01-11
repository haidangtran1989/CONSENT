import random
import os
import ast
from utilities.nlp_utils import typical_name, get_text_with_fullname, replace_name_in_question, split_id

folder_path = "../data/benchmark/gen_conv/"
conv_file_names = os.listdir(folder_path)
for i in range(10000):
    random.shuffle(conv_file_names)

conv_list = list()
for conv_file_name in conv_file_names:
    with open(folder_path + conv_file_name, "r", encoding="utf-8") as test_file:
        content = test_file.read().strip()
        conv_list.append(content)

id_set = set()

with open("../data/conv_test_set.txt", "r", encoding="utf-8") as cur_test_file:
    cur_content = cur_test_file.read().strip()
with open("../data/conv_train_set.txt", "r", encoding="utf-8") as cur_test_file:
    cur_content += "\n" + cur_test_file.read().strip()
cur_lines = cur_content.strip().split("\n")
for cur_line in cur_lines:
    if cur_line.startswith("Q_"):
        q_id, question = split_id(cur_line)
        id_set.add(int(q_id.split("_")[1]))


for conversation in conv_list:
    turns = conversation.split("\n\n")
    news = turns[0]
    turns = turns[1:-1]
    if len(turns) < 4:
        continue
    while True:
        id = random.randint(0, 9999)
        if id not in id_set:
            break
    history = ""
    ok = True
    for turn in turns:
        lines = turn.split("\n")
        entity = lines[3].split("\t")[0]
        fullname = typical_name(entity)
        surname = fullname.split()[-1]
        if history.strip() != "":
            if "'person'" in lines[3] and surname not in history:
                ok = False
                break
            if "'person'" not in lines[3] and fullname not in history:
                ok = False
                break
        history += "\n" + lines[1]
    if not ok:
        continue
    id_set.add(id)
    i = 0
    print("Conversation ID:", id)
    print("News:", news.replace("\n", ": "))
    print()
    for turn in turns:
        lines = turn.split("\n")
        q_id = "Q_" + str(id) + "_" + str(i)
        a_id = "A_" + str(id) + "_" + str(i)
        question = lines[0]
        ans_with_fullname = lines[2]
        entity = lines[3].split("\t")[0]
        entity_name = typical_name(entity)
        firstname = entity_name.split()[0]
        surname = entity_name.split()[-1]
        entity_types_as_str = lines[3].split("\t")[2]
        entity_types = ast.literal_eval(entity_types_as_str)
        if "'person'" in entity_types_as_str:
            if surname in ans_with_fullname:
                ans_with_fullname = ans_with_fullname.replace(entity_name, surname)
                ans_with_fullname = ans_with_fullname.replace(surname, entity_name)
                ans_with_fullname = get_text_with_fullname(ans_with_fullname, entity_name, entity_types)
        can_answer = ans_with_fullname
        if i > 0:
            question = replace_name_in_question(question, entity, entity_types)
        print(q_id + ": " + question)
        if "'person'" in entity_types_as_str:
            if i > 0:
                can_answer = can_answer.replace(entity_name, firstname)
        print(a_id + " with Fullname: " + ans_with_fullname)
        print(a_id + ": " + can_answer)
        print("Entity: " + entity)
        print("Entity Popularity: " + lines[3].split("\t")[1])
        print("Entity Type: " + entity_types_as_str)
        print()
        i += 1
    print("+++++")
