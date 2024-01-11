import torch
from typing import Dict, Tuple
from models.transformer_box_model import TransformerBoxModel
from models.box_decoder import BoxDecoder
from utilities.config import *
from loaders.type_loader import load_types


class BoxModelClassifier(TransformerBoxModel):
    def __init__(self):
        super(BoxModelClassifier, self).__init__()
        self.classifier = BoxDecoder()
        self.types = load_types()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_context_rep = self.build_representation(inputs)
        return self.classifier(mention_context_rep)

    def classify(self, mention_context_list):
        mention_context_rep = self.build_representation_from_texts(mention_context_list)
        probs = self.classifier(mention_context_rep).cpu().data.numpy()
        return [set(self.types[j] for j in range(len(self.types)) if probs[i][j] > TYPE_SIMILARITY_THRESHOLD) for i in range(len(probs))]
