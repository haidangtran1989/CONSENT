import sys
from baselines.sbert.bm25_cross_encoder import sbert_dialog_search, get_sbert_search_print
from entity.type_finder import find_type_list
from loaders.document_loader import get_document
from loaders.title_intro_name_loader import get_full_name_doc_intro, add_names
from utilities.nlp_utils import recognize_mentions, recognize_mentions_in_batch, match_two_list, \
    clean_name, typical_name, split_id, get_acro_name_with_first_letters_of_all_words, \
    get_acro_name_with_first_letters_of_same_word
from .consent_score import get_relevance_score
from .search_result import SearchResult
from utilities.config import *


def select_context_and_entity(dialog_mentions, prev_dialog_turns, prev_dialog_turn_mentions,
                              prev_dialog_turn_mention_context_list, prev_dialog_turn_mention_type_list,
                              question, sbert_sentence_with_context_list, sbert_reranked_results, train_or_test):
    core_results = list()
    max_score = -1000000.0
    max_vars_x = None
    max_vars_y = None
    max_w1 = None
    max_w2 = None
    max_w4 = None
    max_w5 = None
    max_answer_entities = None
    debug = ""
    if len(dialog_mentions) > 0:
        count = 0
        for entities_of_the_same_sentence_with_context in recognize_mentions_in_batch(sbert_sentence_with_context_list):
            news_id = sbert_reranked_results[count].news_id
            entity_mentions_in_an_answer = list()
            entity_mention_context_pairs_in_an_answer = list()
            for entity in entities_of_the_same_sentence_with_context:
                entity_text = clean_name(entity.text)
                news_doc_id = "_".join(news_id.split("_")[:2])
                new_entity_text, answer_context = get_full_name_doc_intro(news_doc_id, entity_text, sbert_sentence_with_context_list[count])
                entity_mentions_in_an_answer.append(new_entity_text)
                entity_mention_context_pairs_in_an_answer.append([new_entity_text, answer_context])
            if not match_two_list(dialog_mentions, entity_mentions_in_an_answer):
                count += 1
                continue
            score, vars_x, vars_y, w1, w2, w4, w5 = get_relevance_score(prev_dialog_turns, prev_dialog_turn_mentions,
                                                                        prev_dialog_turn_mention_type_list, question, sbert_reranked_results[count].sentence, entity_mentions_in_an_answer, train_or_test)
            score += BASE_SCORE_COEFF * sbert_reranked_results[count].sbert_rerank_score
            if w4 is not None:
                w4 = list(w4)
            if w5 is not None:
                w5 = list(w5)
            core_results.append(SearchResult(score, sbert_reranked_results[count].t5_rerank_score, sbert_reranked_results[count].sbert_rerank_score,
                                             sbert_reranked_results[count].context, sbert_reranked_results[count].sentence, news_id, w1, w2, w4, w5, vars_x, vars_y,
                                             entity_mentions_in_an_answer, None))
            if max_score < score:
                max_score = score
                max_vars_x = vars_x
                max_vars_y = vars_y
                max_w1 = w1
                max_w2 = w2
                max_w4 = w4
                max_w5 = w5
                max_answer_entities = entity_mentions_in_an_answer
            count += 1
        core_results.sort(key=lambda result: result.score, reverse=True)
        debug += "Max Score: " + str(max_score) + "\n"
        debug += "Mention-Question Weights: " + str(max_w1) + "\n"
        debug += "Mention-Answer Weights: " + str(max_w2) + "\n"
        debug += "Turn-Question Weights: " + str(max_w4) + "\n"
        debug += "Turn-Answer Weights: " + str(max_w5) + "\n"
        debug += "Max Vars Mention Choice: " + str(max_vars_x) + "\n"
        debug += "Max Vars Turn Choice: " + str(max_vars_y) + "\n"
        debug += "Prev Dialog Mention Context: " + str(prev_dialog_turn_mention_context_list) + "\n"
        debug += "Prev Dialog Turn Mention Types: " + str(prev_dialog_turn_mention_type_list) + "\n"
        debug += "Max Answer Entities: " + str(max_answer_entities) + "\n"
    else:
        core_results = sbert_reranked_results
    return core_results, debug


def predict_entity_types(prev_dialog_turn_mentions, prev_dialog_turns):
    prev_dialog_turn_mention_type_list = list()
    prev_dialog_turn_mention_context_list = list()
    for k in range(len(prev_dialog_turn_mentions)):
        current_turn_mention_contexts = list()
        for l in range(len(prev_dialog_turn_mentions[k])):
            dialog_entity_text, dialog_entity_context = get_full_name_doc_intro(None, prev_dialog_turn_mentions[k][l], prev_dialog_turns[k])
            if prev_dialog_turn_mentions[k][l] != dialog_entity_text:
                prev_dialog_turn_mentions[k][l] = dialog_entity_text
            current_turn_mention_contexts.append([dialog_entity_text, dialog_entity_context])
        prev_dialog_turn_mention_type_list.append(find_type_list(current_turn_mention_contexts))
        prev_dialog_turn_mention_context_list.append(current_turn_mention_contexts)
    return prev_dialog_turn_mention_context_list, prev_dialog_turn_mention_type_list


def dialog_search(prev_dialog_turns, question_with_id, dependency_turn, dependency_entity, golden_answer, train_or_test):
    question_id = question_with_id[0]
    question = question_with_id[1]
    history = "\n".join(prev_dialog_turns)
    history_mentions = list(set(clean_name(mention.text) for mention in recognize_mentions(history)))
    prev_dialog_turn_mentions = list()
    firstname_to_fullname = dict()
    acro_non_person_to_fullname = dict()
    acro_person_to_fullname = dict()
    prev_questions = list()
    prev_answers = list()
    for prev_turn in prev_dialog_turns:
        lines = prev_turn.split("\n")
        prev_questions.append(lines[0])
        prev_answers.append(lines[1])
        turn_mentions = set(clean_name(mention.text) for mention in recognize_mentions(prev_turn))
        for turn_mention in turn_mentions:
            firstname = turn_mention.split()[0]
            if len(firstname) < len(turn_mention):
                firstname_to_fullname[firstname] = turn_mention
            acro_non_person_name = get_acro_name_with_first_letters_of_all_words(turn_mention)
            if acro_non_person_name is not None and len(acro_non_person_name) > 1:
                acro_non_person_to_fullname[acro_non_person_name] = turn_mention
            acro_person_name = get_acro_name_with_first_letters_of_same_word(firstname)
            if acro_person_name is not None and len(acro_person_name) > 1:
                acro_person_to_fullname[acro_person_name] = turn_mention
        new_turn_mentions = set()
        for turn_mention in turn_mentions:
            if turn_mention in firstname_to_fullname:
                new_turn_mentions.add(firstname_to_fullname[turn_mention])
            elif turn_mention in acro_person_to_fullname:
                new_turn_mentions.add(acro_person_to_fullname[turn_mention])
            elif turn_mention in acro_non_person_to_fullname:
                new_turn_mentions.add(acro_non_person_to_fullname[turn_mention])
            else:
                new_turn_mentions.add(turn_mention)
        turn_mentions = new_turn_mentions
        prev_dialog_turn_mentions.append(list(turn_mentions))

    prev_dialog_turn_mention_context_list, prev_dialog_turn_mention_type_list = predict_entity_types(prev_dialog_turn_mentions, prev_dialog_turns)

    # BM25 + SBERT
    enriched_question, sbert_reranked_results, sbert_sentence_with_context_list = sbert_dialog_search(history_mentions, question, golden_answer, train_or_test)
    sbert_search_print = get_sbert_search_print(enriched_question, sbert_reranked_results, question_with_id)
    new_sbert_reranked_results = list()
    for result in sbert_reranked_results:
        if result.sbert_rerank_score > MIN_BASE_SCORE or result.news_id == "N/A":
            new_sbert_reranked_results.append(result)
    sbert_reranked_results = new_sbert_reranked_results[:TOP_S]
    sbert_sentence_with_context_list = [result.context for result in sbert_reranked_results]
    saved_sbert_search_print = sbert_search_print
    documents = []
    for doc in sbert_reranked_results:
        doc_id = "_".join(doc.news_id.split("_")[:2])
        doc_content = get_document(doc_id)
        if doc_content is not None:
            documents.append(doc_content)
    add_names(documents)
    if len(prev_dialog_turns) == 0 or len(sbert_reranked_results) == 0:
        return sbert_search_print
    console_print = sbert_search_print + "\n\nTo be continued...\n"

    core_results, debug = select_context_and_entity(history_mentions, prev_dialog_turns, prev_dialog_turn_mentions,
                                                    prev_dialog_turn_mention_context_list, prev_dialog_turn_mention_type_list,
                                                    question, sbert_sentence_with_context_list, sbert_reranked_results, train_or_test)
    print("History: " + str(prev_dialog_turns))
    print("Dependency Turn: " + dependency_turn)
    print("Dependency Entity: " + dependency_entity)
    print("Prev Dialog Mentions: " + str(prev_dialog_turn_mentions))
    print(debug)
    console_print += "\n" + "History: " + str(prev_dialog_turns)
    console_print += "\n" + "Dependency Turn: " + dependency_turn
    console_print += "\n" + "Dependency Entity: " + dependency_entity
    console_print += "\n" + "Prev Dialog Mentions: " + str(prev_dialog_turn_mentions)
    console_print += "\n" + debug + "\n"
    if len(core_results) == 0:
        return saved_sbert_search_print
    for result in core_results:
        current_sentence = result.sentence
        console_print_line = str(result.score) + "\t" + str(result.t5_rerank_score) + "\t" + str(result.sbert_rerank_score) + "\t" + current_sentence.strip() +\
                             "\t<--" + result.context.strip() + "\t" + result.news_id + "\t" + question_id + "\n" +\
                             "Mention to question weight: " + str(result.mention_to_question_weight) + "\n" +\
                             "Mention to answer weight: " + str(result.mention_to_answer_weight) + "\n" +\
                             "Turn to question weight: " + str(result.turn_to_question_weight) + "\n" +\
                             "Turn to answer weight: " + str(result.turn_to_answer_weight) + "\n" +\
                             "Mention choice: " + str(result.mention_choice) + "\n" +\
                             "Turn choice: " + str(result.turn_choice) + "\n" + \
                             "Answer mentions: " + str(result.answer_mentions) + "\n" + \
                             "Answer mention types: " + str(result.answer_mention_types) + "\n---"
        print(console_print_line)
        console_print += "\n" + console_print_line
    return console_print


def search_from_questions(questions, answers, dependency_turns, dependency_entities, golden_answers, train_or_test):
    prev_dialog_turns = list()
    total_print = ""
    for i in range(len(questions)):
        console_print = dialog_search(prev_dialog_turns, questions[i], dependency_turns[i], dependency_entities[i], golden_answers[i], train_or_test)
        total_print += "\n\n" + console_print + "\n+++++\n"
        prev_dialog_turns.append(questions[i][1] + "\n" + answers[i])
    return total_print


def search_from_testset(from_conversation, to_conversation, train_or_test):
    if train_or_test == "test":
        test_file = open("../data/conv_test_set.txt", "r", encoding="utf-8")
    else:
        test_file = open("../data/conv_train_set.txt", "r", encoding="utf-8")
    content = test_file.read()
    conversations = content.split("+++++")
    cnt = 0
    for conversation in conversations:
        conversation = conversation.strip()
        if conversation == "":
            continue
        turns = conversation.split("\n\n")
        _, conversation_id = split_id(turns[0].split("\n")[0])
        if from_conversation <= cnt <= to_conversation:
            questions = list()
            golden_answers = list()
            answers = list()
            dependency_turns = list()
            dependency_entities = list()
            for i in range(1, len(turns)):
                turn = turns[i]
                lines = turn.split("\n")
                question_id, question = split_id(lines[0])
                canonical_answer_id_with_fullname, canonical_answer_with_fullname = split_id(lines[1])
                canonical_answer_id, canonical_answer = split_id(lines[2])
                dependency_entity = typical_name(lines[3][len("Entity: "):])
                questions.append((question_id, question))
                golden_answers.append(canonical_answer_with_fullname)
                answers.append(canonical_answer)
                dependency_turns.append("")
                dependency_entities.append(dependency_entity)
            total_print = search_from_questions(questions, answers, dependency_turns, dependency_entities, golden_answers, train_or_test)
            if train_or_test == "test":
                ret_file = open("../data/result/consent_test/" + str(conversation_id) + ".txt", "w", encoding="utf-8")
            else:
                ret_file = open("../data/result/consent_train/" + str(conversation_id) + ".txt", "w", encoding="utf-8")
            ret_file.write(total_print)
            ret_file.close()
        cnt += 1
    test_file.close()


if __name__ == "__main__":
    from_conversation = int(sys.argv[1])
    to_conversation = int(sys.argv[2])
    train_or_test = sys.argv[3]
    search_from_testset(from_conversation, to_conversation, train_or_test)
