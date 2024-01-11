from math import exp
import spacy

SELECTED_ENTITY_TYPES = set(['EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'])
WH_WORDS = ["what", "where", "when", "how", "whom", "which", "who"]
POTENTIAL_NOUN_PREFIX = ["his", "her", "their", "its", "my", "our", "your"]
COREFERENCES = ["his", "her", "their", "its", "my", "our", "he", "she", "they", "it", "i", "we", "you"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

nlp_sm = spacy.load("en_core_web_sm")
nlp_sm.enable_pipe("senter")


def logit_to_prob(question_to_answer_base_score):
    odd = exp(question_to_answer_base_score)
    question_to_answer_base_score = odd / (1 + odd)
    return question_to_answer_base_score


def inside(small_name, long_name):
    words = set(long_name.strip().lower().split())
    small_words = set(small_name.strip().lower().split())
    for word in small_words:
        if word not in words:
            return False
    return True


def get_text_with_fullname(text, entity_mention, entity_types):
    if "person" not in entity_types:
        return text
    text = " " + text + " "
    names = set([x.text for x in recognize_mentions(text)])
    for name in names:
        if inside(name, entity_mention) or inside(entity_mention, name):
            if entity_mention != name:
                text = text.replace(" " + name + " ", " " + entity_mention + " ")
                return text.strip()
    return text.replace("  ", " ").strip()


def get_acro_name_with_first_letters_of_all_words(fullname):
    parts = fullname.split()
    ret = ""
    for part in parts:
        if not part[0].isupper():
            return None
        ret += part[0]
    return ret


def get_acro_name_with_first_letters_of_same_word(word_in_fullname):
    if len(word_in_fullname) <= 5:
        return word_in_fullname
    return word_in_fullname[:4]


def remove_too_specific_date_in_question(question):
    for i in range(1, 32):
        for month in MONTHS:
            span = "on " + str(i) + " " + month
            if span in question:
                return question.replace(span, "in " + month)
            span = "On " + str(i) + " " + month
            if span in question:
                return question.replace(span, "In " + month)
    return question


def replace_name_in_question(question, entity, types):
    name = typical_name(entity)
    new_mention = name
    if "person" in types:
        new_mention = get_acro_name_with_first_letters_of_same_word(name.split()[0])
    elif "company" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " company"
    elif "organization" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " org"
    elif "film" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " film"
    elif "law" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " law"
    elif "area" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " area"
    elif "song" in types:
        new_mention = get_acro_name_with_first_letters_of_all_words(name) + " song"
    else:
        new_mention = get_acro_name_with_first_letters_of_all_words(name)
    return question.replace(name, new_mention)


def split_id(text):
    pos = text.find(": ")
    return text[:pos], text[pos + 2:]


def clean_name(name):
    name = name.replace("-", " ")
    if name.endswith("'s"):
        name = name[:-2]
    if name.startswith("a "):
        name = name[len("a "):]
    if name.startswith("the "):
        name = name[len("the "):]
    return name


def typical_name(entity):
    name = entity.replace("_", " ")
    if " (" in name:
        name = name[:name.index(" (")]
    return name


def match_two_list(list1, list2):
    for name1 in list1:
        if match(name1, list2):
            return True
    return False


def possible_name_match(name1, name2):
    parts1 = set(name1.split(" "))
    parts2 = set(name2.split(" "))
    if get_acro_name_with_first_letters_of_same_word(name1.split()[0]) == name2:
        return True
    if get_acro_name_with_first_letters_of_same_word(name2.split()[0]) == name1:
        return True
    if get_acro_name_with_first_letters_of_all_words(name1) == name2:
        return True
    if get_acro_name_with_first_letters_of_all_words(name2) == name1:
        return True
    return len(parts1.intersection(parts2)) > 0


def match(mention, other_mentions):
    if mention.lower() in COREFERENCES:
        return True
    for another_mention in other_mentions:
        if mention in another_mention or another_mention in mention:
            return True
    return False


def recognize_mentions(text):
    return recognize_mentions_in_batch([text])[0]


def recognize_mentions_in_batch(texts):
    mentions_in_batch = list()
    for doc in nlp_sm.pipe(texts):
        mentions_in_batch.append(set([ent for ent in doc.ents if ent.label_ in SELECTED_ENTITY_TYPES]))
    return mentions_in_batch


def split_sentences(text):
    return split_sentences_in_batch([text])[0]


def split_sentences_in_batch(texts):
    sentences_in_batch = list()
    for doc in nlp_sm.pipe(texts):
        sentences_in_batch.append(list([sent.text for sent in doc.sents]))
    return sentences_in_batch
