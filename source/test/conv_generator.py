import random
import requests
import re
import time
import html
import gzip
from entity.type_finder import find_type_list
from test.question_generator import generate_best_question
from utilities.config import MENTION_ENTITY_FILE_PATH
from utilities.nlp_utils import split_sentences, typical_name, get_text_with_fullname, remove_too_specific_date_in_question
from urllib.parse import unquote
from utilities.text_text_relatedness_calculator import get_text_to_text_related_score

entity_popularity = dict()
with gzip.open(MENTION_ENTITY_FILE_PATH, "rt", encoding="utf-8") as mention_entity_file:
    for line in mention_entity_file:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        freq = int(parts[2])
        cur_total_freq = entity_popularity.get(parts[1], 0) + freq
        entity_popularity[parts[1]] = cur_total_freq


def extract_entities_from_wiki_tag(text):
    entity_to_mentions = dict()
    i = 0
    while True:
        i = text.find("href=\"/wiki/", i)
        if i < 0:
            break
        i += len("href=\"/wiki/")
        j = text.find("\"", i)
        entity = unquote(text[i:j])
        i = text.find(">", i)
        if i < 0:
            break
        i += 1
        j = text.find("<", i)
        mention = text[i:j]
        if mention != "" and mention[0].isupper() and mention == typical_name(entity):
            entity_to_mentions[entity] = mention
        i = j
    clean_text = re.sub("<.*?>", "", text)
    return clean_text, entity_to_mentions


def extract_summaries(link):
    html_content = requests.get(link).text
    html_content = html_content.split(">Current events by month<")[0]
    time.sleep(1)
    daily_news = html_content.split("<div class=\"current-events-content description\">")[1:]
    text_list = list()
    ret = list()
    for cur_news in daily_news:
        parts = cur_news.split("<div class=\"current-events-content-heading\"")
        for part in parts:
            if "id=\"Ongoing_events\"" in part:
                part = part.split("id=\"Ongoing_events\"")[0]
            if "<span class=\"summary\">" in part:
                part = part.split("<span class=\"summary\">")[0]
            if "<style data" in part:
                part = part.split("<style data")[0]
            part = part.strip()
            if part == "":
                continue
            lines = part.split("<li>")[1:]
            for line in lines:
                line = html.unescape(line).strip()
                text = re.sub("<.*?>", "", line).strip()
                text_list.append((text, line))
                if "</ul>" in line:
                    if len(text_list) > 1:
                        if not text_list[0][0].endswith(")"):
                            if text_list[-1][0].endswith(")"):
                                for k in range(1, len(text_list)):
                                    _, entity_to_mentions = extract_entities_from_wiki_tag(text_list[k][1])
                                    ret.append((text_list[0][0], text_list[k][0], entity_to_mentions))
                    text_list = list()
    return ret


selected_sentences_cache = dict()


def entity_should_be_focused(entity, types, context, max_pop):
    entity_name = typical_name(entity)
    if not entity_name.isascii():
        return False
    if "#" in entity_name or len(entity_name.split()) == 1:
        return False
    words = entity_name.split()
    for word in words:
        if word[0].islower() or not word[0].isalpha():
            return False
    pop = entity_popularity.get(entity, -1)
    if pop <= max_pop and " " in entity_name:
        return True
    return False


def select_sentences(entity, entity_types, turn_number):
    if entity in selected_sentences_cache:
        return selected_sentences_cache[entity]
    pop = entity_popularity.get(entity, -1)
    entity_name = typical_name(entity)
    link = "https://en.wikipedia.org/wiki/" + entity
    html_content = requests.get(link).text
    time.sleep(1)
    html_content_parts = re.split('<div id="toc".*?>', html_content)
    text = html_content
    if len(html_content_parts) >= 2:
        text = html_content_parts[1]
        if "</ul>" not in text:
            return list()
        text = text[text.find("</ul>"):]
    text = re.split('<ol class="references">', text)[0]
    paragraphs = text.split("<p>")[1:]
    result = list()
    for para in paragraphs:
        para = para.split("</p>")[0]
        para, entity_to_mentions_in_para = extract_entities_from_wiki_tag(para)
        para = html.unescape(para).strip()
        para = re.sub("\[\d+\]", "", para)
        para = para.replace("[citation needed]", "")
        para = para.replace("[update]", "")
        word_num = len(para.split(" "))
        if word_num < 15:
            continue
        if "\n" in para or "2019" in para or "2020" in para or "2021" in para or "2022" in para or "2023" in para:
            continue
        sentences = split_sentences(para)
        sentences_picked = list()
        for sent in sentences:
            if ("2018" not in sent) and ("2017" not in sent):
                if pop < 50:
                    continue
                elif random.randint(0, 1) == 0:
                    continue
            if "{" in sent or "}" in sent or "+" in sent:
                continue
            ent_to_ment_in_sent = dict()
            contains_other_entity = False
            for cur_ent, cur_ment in entity_to_mentions_in_para.items():
                if cur_ment in sent:
                    types = find_type_list([[cur_ment, sent]])[0]
                    if cur_ent != entity and entity_should_be_focused(cur_ent, types, sent, 100):
                        ent_to_ment_in_sent[cur_ent] = (cur_ment, types)
                        contains_other_entity = True
            if turn_number < 3 and not contains_other_entity:
                continue
            if "person" in entity_types:
                if entity_name.split()[-1] not in sent:
                    continue
            else:
                if entity_name not in sent:
                    continue
            if len(sent.strip().split()) < 15:
                continue
            if sent.strip().endswith(":"):
                continue
            sentences_picked.append((sent.strip(), ent_to_ment_in_sent))
        result.extend(sentences_picked)
    selected_sentences_cache[entity] = result
    return result


def generate_turns_from_news(cat, summary, ent_to_mentions):
    while summary.strip().endswith(")") or summary.strip().endswith("),"):
        idx = summary.rfind("(")
        if idx >= 0:
            summary = summary[:idx].strip()
        else:
            break
    filtered_entities = list()
    for ent, mention in ent_to_mentions.items():
        ent_types = find_type_list([[typical_name(ent), summary]])[0]
        if entity_should_be_focused(ent, ent_types, summary, 500):
            filtered_entities.append((ent, mention, entity_popularity.get(ent, -1), ent_types))
    if len(filtered_entities) == 0:
        return list()
    random_idx = random.randint(0, len(filtered_entities) - 1)
    extracted_seed_entity, extracted_mention, popularity, extracted_entity_types = filtered_entities[random_idx]
    last_answer = summary
    selected_answers = set()
    turn_with_info_list = list()
    all_entities = {extracted_seed_entity}
    all_entities_with_infos = [(extracted_seed_entity, extracted_entity_types)]
    # This is for each turn
    for k in range(6):
        follow = False
        # Few times to try for current entity
        for j in range(3):
            random_idx = random.randint(0, len(all_entities_with_infos) - 1)
            if k >= 3 and random.randint(0, 1) == 0 and not follow:
                random_idx = len(all_entities_with_infos) - 1
                follow = True
            current_entity, current_entity_types = all_entities_with_infos[random_idx]
            current_entity_name = typical_name(current_entity)
            if k >= 3 and current_entity_name in last_answer:
                if random.randint(0, 1) == 0:
                    continue
            if k >= 2 and current_entity == extracted_seed_entity:
                continue
            sentences = select_sentences(current_entity, current_entity_types, k)
            if len(sentences) == 0:
                continue
            sent_texts = [x[0] for x in sentences]
            scores_with_last_answer = get_text_to_text_related_score(last_answer, sent_texts)
            sentences_with_scores = [(scores_with_last_answer[i], sent_texts[i], sentences[i][1]) for i in range(len(sentences))]
            sentences_with_scores.sort(key=lambda sent: sent[0], reverse=True)
            sentences_with_scores = sentences_with_scores[:5]
            filtered_sentences = list()
            for i in range(len(sentences_with_scores)):
                if sentences_with_scores[i][1] not in selected_answers:
                    filtered_sentences.append(sentences_with_scores[i])
            if len(filtered_sentences) == 0:
                continue
            random_idx = random.randint(0, len(filtered_sentences) - 1)
            last_answer = filtered_sentences[random_idx][1]
            last_answer_with_fullname = get_text_with_fullname(last_answer, current_entity_name, current_entity_types)
            best_question = generate_best_question(cat + ": " + summary, turn_with_info_list, last_answer_with_fullname, current_entity)
            if best_question is None:
                continue
            if k > 0:
                best_question = remove_too_specific_date_in_question(best_question)
            ent_map = filtered_sentences[random_idx][2]
            selected_answers.add(last_answer)
            turn_with_info_list.append((last_answer, last_answer_with_fullname, current_entity, current_entity_types, best_question))
            for cur_ent, cur_ment_and_types in ent_map.items():
                if cur_ent not in all_entities:
                    all_entities.add(cur_ent)
                    all_entities_with_infos.append((cur_ent, cur_ment_and_types[1]))
            break
    return turn_with_info_list


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


if __name__ == "__main__":
    for month in MONTHS:
        link = "https://en.wikipedia.org/wiki/Portal:Current_events/" + month + "_2018"
        summaries = extract_summaries(link)
        for (cat, summary, ent_to_mentions) in summaries:
            generated_turns = generate_turns_from_news(cat, summary, ent_to_mentions)
            if len(generated_turns) < 3:
                continue
            file_name = time.strftime("%Y-%m-%d-%H-%M-%S")
            with open("../data/benchmark/gen_conv/" + file_name + ".txt", "wt", encoding="utf-8") as out_file:
                out_file.write(cat + "\n")
                out_file.write(summary + "\n\n")
                out_file.write("\n\n".join([(x[-1] + "\n" + x[0] + "\n" + x[1] + "\n" + x[2] + "\t" + str(entity_popularity.get(x[2], -1)) + "\t" + str(x[3])) for x in generated_turns]) + "\n")
                out_file.write("\n" + link + "\n")
