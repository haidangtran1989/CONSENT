import openai
import time
from utilities.nlp_utils import recognize_mentions, typical_name
from utilities.text_text_relatedness_calculator import get_text_to_text_related_score

QUESTION_PREFIXES = [
    "Spoken Short Question",
    "More Spoken Short Question without subject: "
]

INITIAL_PROMPT = """
Context: Shortly after his allegations against Zolotov, Alexei Navalny was imprisoned for staging protests in January 2018.
Spoken Short Question containing Alexei Navalny: Why Alexei Navalny got collared?
More Spoken Short Question without subject: Why got collared?

Context: After not getting out of his group at the Grand Slam, Dave Chisnall won through to the final of the Players Championship Finals, surviving three match darts from Jelle Klaasen along the way.
Spoken Short Question containing Dave Chisnall: Road to Players Championship Finals for Dave Chisnall?
More Spoken Short Question without subject: Road to Players Championship Finals?

Context: Before that, Boyko Borisov had accepted the resignation of Finance Minister Simeon Djankov after a dispute over farm subsidies and promised a cut in power prices and punishing foreign-owned companies—a potential risk in damaging Bulgaria-Czech Republic relations—but protests continued.
Spoken Short Question containing Boyko Borisov: What Boyko Borisov pledged to ease protests?
More Spoken Short Question without subject: Pledged what to ease protests?

Context: The next day, Shahid Khaqan Abbasi chaired a meeting of the NSC which expressed disappointment over Trump remarks and observed that Pakistan cannot be held responsible for US failure in Afghanistan.
Spoken Short Question containing Shahid Khaqan Abbasi: What Shahid Khaqan Abbasi led at NSC meeting?
More Spoken Short Question without subject: Led what at NSC meeting?

Context: Cheddar Man was relatively small compared to modern Europeans, with an estimated stature of around 1.66 metres (5 ft 5 in), and weighing around 66 kilograms (146 lb).
Spoken Short Question containing Cheddar Man: How tall Cheddar Man was?
More Spoken Short Question without subject: How tall?

"""

openai.api_key = ""


def generate_text(prompt):
    ret = ""
    for i in range(3):
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=1.0,
                max_tokens=16,
                top_p=1.0
            )
            choice = response.choices[0]
            ret = choice.text
            time.sleep(0.1)
            break
        except:
            print("error calling OpenAI API")
            time.sleep(5)
            continue
    ret = ret.replace("\n", " ").strip()
    if "Spoken Short Question" in ret:
        ret = ret[:ret.find("Spoken Short Question")].strip()
    elif "More Spoken Short Question" in ret:
        ret = ret[:ret.find("More Spoken Short Question")].strip()
    if not ret.endswith("?"):
        ret += "?"
    ret = ret[:ret.find("?")].strip() + "?"
    return ret.strip()


MAX_TRIES = 5


def choose_best_question(candidates, answer):
    best_question = None
    max_score = -1
    candidate_info = list()
    for question in candidates:
        if len(question) < 5 or len(question.split()) > 10:
            continue
        sim_list = get_text_to_text_related_score(question, [answer])
        rel_with_answer = sim_list[0]
        candidate_info.append("Candidate question: {} (rel score {} with current answer)".format(question, rel_with_answer))
        score = rel_with_answer
        if max_score < score:
            max_score = score
            best_question = question
    return best_question, candidate_info


def contain_out_of_blue_name(question, history):
    names = [mention.text for mention in recognize_mentions(question)]
    for name in names:
        if name not in history:
            return True
    return False


def generate_questions(answer, fullname, have_name, history):
    candidates = []
    for i in range(MAX_TRIES):
        prompt = INITIAL_PROMPT + "Context: " + answer + "\n"
        for j in range(len(QUESTION_PREFIXES)):
            if j < 1:
                prompt += QUESTION_PREFIXES[j] + " containing " + fullname + ": "
            else:
                prompt += QUESTION_PREFIXES[j]
            q = generate_text(prompt)
            if have_name and j == 0:
                candidates.append(q)
            if (not have_name) and (j == 1):
                candidates.append(q)
            prompt += q + "\n"
    filtered_candidates = [q for q in candidates if not contain_out_of_blue_name(q, history)]
    if len(filtered_candidates) > 0:
        return filtered_candidates
    return candidates


def generate_best_question(news, turn_with_info_list, last_answer_with_fullname, current_entity):
    entity_name = typical_name(current_entity)
    history = news + "\n\n" + "\n\n".join(turn[0] + "\n" + turn[-1] for turn in turn_with_info_list)
    have_name = True
    if len(turn_with_info_list) > 0 and current_entity == turn_with_info_list[-1][2]:
        have_name = False
    candidates = generate_questions(last_answer_with_fullname, entity_name, have_name, history)
    best_question, candidate_info = choose_best_question(candidates, last_answer_with_fullname)
    return best_question
