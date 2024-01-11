import time
import gzip
from utilities.config import *

mention_entity_freq = dict()
mention_entity_freq_prob = dict()
start = time.time()
with gzip.open(MENTION_ENTITY_FILE_PATH, "rt", encoding="utf-8") as mention_entity_file:
    for line in mention_entity_file:
        parts = line.strip().split("\t")
        freqs = mention_entity_freq.get(parts[0], dict())
        freqs[parts[1]] = int(parts[2])
        mention_entity_freq[parts[0]] = freqs

for mention, freqs in mention_entity_freq.items():
    sum_freq = 0
    for entity, freq in freqs.items():
        sum_freq += freq
    freq_probs = dict()
    for entity, freq in freqs.items():
        freq_probs[entity] = 1.0 * freq, 1.0 * freq / sum_freq
    mention_entity_freq_prob[mention] = freq_probs
end = time.time()
print("Loaded mention entity freq and prob for PEL with " + str(end - start))


def get_entities(mention: str) -> dict:
    return mention_entity_freq_prob.get(mention, dict())


if __name__ == "__main__":
    entities = get_entities("trump")
    for name, freq_prob in entities.items():
        print(name, "-->", freq_prob)
