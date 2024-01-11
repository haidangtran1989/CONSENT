import gzip
from utilities.nlp_utils import typical_name

entity_to_title_intro = dict()
with gzip.open("../data/resource/wiki-intro.tsv.gz", "rt", encoding="utf-8") as intro_file:
    for line in intro_file:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            entity_to_title_intro[parts[0]] = typical_name(parts[1]) + "\t" + parts[2]
print("Done with loading Wikipedia introduction texts!")


def get_title_intro(entity):
    return entity_to_title_intro.get(entity, None)


if __name__ == "__main__":
    print(get_title_intro("Manchester_United_F.C."))
    print(get_title_intro("Donald_Trump"))
