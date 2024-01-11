from utilities.config import *
from models.box_model_classifier import BoxModelClassifier
import time

FINE_GRAIN_SIMILARITY = 0.5

start = time.time()
model = BoxModelClassifier()
checkpoint = torch.load(BOX_TYPE_MODEL_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.classifier.init_type_box()
model.to(DEVICE)
end = time.time()
print("Done with loading box type model with " + str(end - start))


def find_type_list(mention_context_list):
    size = len(mention_context_list)
    if size == 0:
        return list()
    current_mention_context_list = list()
    total_type_list = list()
    for i in range(size):
        current_mention_context_list.append(mention_context_list[i])
        if (i + 1 == size) or ((i + 1) % TYPE_LIST_BATCH == 0):
            total_type_list.extend(model.classify(current_mention_context_list))
            current_mention_context_list = list()
    return total_type_list
