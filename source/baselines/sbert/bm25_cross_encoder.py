import json
from pyserini.search import LuceneSearcher
from search.search_result import SearchResult
from utilities.config import *
from utilities.nlp_utils import split_sentences_in_batch
from utilities.text_text_relatedness_calculator import get_question_to_answer_related_score

lucene_searcher = LuceneSearcher('../data/resource/news-wiki-en-index')
print("Loaded BM25 Lucene index!")


def split_sentences_of_documents(question, top_search_documents, top_search_ids, golden_answer, train_or_test):
    sentence_with_context_list = list()
    question_and_sentence_pairs = list()
    if train_or_test == "train":
        question_and_sentence_pairs.append((question, golden_answer))
        sentence_with_context_list.append((golden_answer, "N/A"))
    k = 0
    for sentences_of_a_document in split_sentences_in_batch(top_search_documents):
        for i in range(len(sentences_of_a_document)):
            question_and_sentence_pairs.append((question, sentences_of_a_document[i]))
            sentence_with_context_list.append((sentences_of_a_document[i], top_search_ids[k] + "_" + str(i)))
        k += 1
    return sentence_with_context_list, question_and_sentence_pairs


def sbert_cross_encoder_rerank(question_and_sentence_pairs, sentence_with_context_list, train_or_test):
    sbert_reranked_results = list()
    if len(question_and_sentence_pairs) > 0:
        ts = list()
        score_with_question = list()
        for i in range(len(question_and_sentence_pairs)):
            ts.append(question_and_sentence_pairs[i])
            if i + 1 == len(question_and_sentence_pairs) or (i + 1) % 128 == 0:
                score_with_question.extend(get_question_to_answer_related_score(ts))
                ts = list()
        for i in range(len(question_and_sentence_pairs)):
            news_sent_id = sentence_with_context_list[i][1]
            sbert_reranked_results.append(SearchResult(None, None, score_with_question[i],
                    sentence_with_context_list[i][0], question_and_sentence_pairs[i][1], news_sent_id, None, None, None, None, None, None, None, None))
    sbert_reranked_results.sort(key=lambda result: result.sbert_rerank_score, reverse=True)
    if train_or_test == "train":
        for i in range(len(sbert_reranked_results)):
            if sbert_reranked_results[i].news_id == "N/A":
                new_results = [sbert_reranked_results[i]] + sbert_reranked_results[:i] + sbert_reranked_results[i + 1:]
                sbert_reranked_results = new_results
                break
    sbert_sentence_with_context_list = [result.context for result in sbert_reranked_results]
    return sbert_reranked_results, sbert_sentence_with_context_list


def sbert_dialog_search(history_mentions, question, golden_answer, train_or_test):
    enriched_question = " ".join(history_mentions) + " " + question
    bm25_search_results = lucene_searcher.search(enriched_question, TOP_D)
    top_search_documents = list()
    top_search_ids = list()
    for top_result in bm25_search_results:
        doc_json = json.loads(top_result.raw)
        top_search_documents.append(doc_json["contents"])
        top_search_ids.append(doc_json["id"])
    sentence_with_context_list, question_and_sentence_pairs = split_sentences_of_documents(question,
            top_search_documents, top_search_ids, golden_answer, train_or_test)
    sbert_reranked_results, sbert_sentence_with_context_list = sbert_cross_encoder_rerank(question_and_sentence_pairs, sentence_with_context_list, train_or_test)
    return enriched_question, sbert_reranked_results, sbert_sentence_with_context_list


def get_sbert_search_print(enriched_question, sbert_reranked_results, question_with_id):
    console_print = "Expanded question: '" + enriched_question + "'"
    console_print += "\n" + "Question: " + question_with_id[1]
    for i in range(min(len(sbert_reranked_results), TOP_S)):
        news_id = sbert_reranked_results[i].news_id
        console_print_line = str(sbert_reranked_results[i].sbert_rerank_score) + "\t" + sbert_reranked_results[i].sentence.strip() +\
                             "\t<--" + sbert_reranked_results[i].context.strip() + "\t" + str(news_id) + "\t" + question_with_id[0]
        console_print += "\n" + console_print_line
    return console_print
