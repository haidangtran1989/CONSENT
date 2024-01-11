import gurobipy as gp
from gurobipy import GRB, quicksum
from utilities.config import *
from utilities.nlp_utils import get_acro_name_with_first_letters_of_same_word, \
    get_acro_name_with_first_letters_of_all_words
from utilities.text_text_relatedness_calculator import get_text_to_text_related_score


def get_relevance_score(prev_dialog_turns, prev_dialog_turn_mentions, prev_dialog_turn_mention_type_list, question, answer, answer_mentions, train_or_test):
    search_model = gp.Model("dialog_search")
    mention_to_question_weight = list()
    mention_to_answer_weight = list()
    turn_to_question_weight = get_text_to_text_related_score(question, prev_dialog_turns)
    turn_to_answer_weight = get_text_to_text_related_score(answer, prev_dialog_turns)
    x = list()
    y = list()
    mention_size = 0
    turn_size = len(prev_dialog_turns)
    for k in range(turn_size):
        y.append(search_model.addVar(vtype=GRB.BINARY, name="y_" + str(k)))
        if len(prev_dialog_turn_mentions[k]) == 0:
            continue
        for i in range(len(prev_dialog_turn_mentions[k])):
            entity_types_verbalized = prev_dialog_turn_mentions[k][i] + " is " + ", ".join(list(prev_dialog_turn_mention_type_list[k][i]))
            new_question = question
            acro_first = get_acro_name_with_first_letters_of_same_word(prev_dialog_turn_mentions[k][i].split()[0])
            acro_full = get_acro_name_with_first_letters_of_all_words(prev_dialog_turn_mentions[k][i])
            if acro_first in new_question:
                new_question = new_question.replace(acro_first, prev_dialog_turn_mentions[k][i])
            elif acro_full is not None and len(acro_full) > 1 and acro_full in new_question:
                new_question = new_question.replace(acro_full, prev_dialog_turn_mentions[k][i])
            new_answer = answer
            if prev_dialog_turn_mentions[k][i] in answer_mentions:
                new_answer = prev_dialog_turn_mentions[k][i] + ": " + answer
            rel_entity_text = get_text_to_text_related_score(entity_types_verbalized, [new_question, new_answer])
            mention_to_question_weight.append(rel_entity_text[0])
            mention_to_answer_weight.append(rel_entity_text[1])
            x.append(search_model.addVar(vtype=GRB.BINARY, name="x_" + str(mention_size)))
            search_model.addConstr(x[mention_size] <= y[k], "c_" + str(mention_size))
            mention_size += 1
    vars_x = None
    vars_y = None
    obj_val = 0
    if train_or_test == "test":
        search_model.setObjective(
            quicksum((ENTITY_QUESTION_COEFF * mention_to_question_weight[i] + ENTITY_ANSWER_COEFF * mention_to_answer_weight[i] - ENTITY_PARSIMONY_COEFF) * x[i] for i in range(mention_size)) +
            quicksum((TURN_QUESTION_COEFF * turn_to_question_weight[k] + TURN_ANSWER_COEFF * turn_to_answer_weight[k] - TURN_PARSIMONY_COEFF) * y[k] for k in range(turn_size)), GRB.MAXIMIZE)
        if MAX_RELEVANT_TURNS_OR_ENTITIES is not None:
            search_model.addConstr(quicksum(x[i] for i in range(mention_size)) <= MAX_RELEVANT_TURNS_OR_ENTITIES, "c_top_entity")
            search_model.addConstr(quicksum(y[i] for i in range(turn_size)) <= MAX_RELEVANT_TURNS_OR_ENTITIES, "c_top_turn")
        search_model.optimize()
        vars_x = [v.X for v in search_model.getVars() if "x_" in v.VarName]
        vars_y = [v.X for v in search_model.getVars() if "y_" in v.VarName]
        obj_val = search_model.ObjVal
    return obj_val, vars_x, vars_y, mention_to_question_weight, mention_to_answer_weight,\
            turn_to_question_weight, turn_to_answer_weight
