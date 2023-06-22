from app.rankers.flan_t5_ranker import FlanT5Sacc
from app.rankers.googletrans_wrapper import GoogleTransWrapper


class FlanT5SaccPL(FlanT5Sacc):
    def __init__(self, size: str, device) -> None:
        super().__init__(size, device)
        self.googletrans: GoogleTransWrapper = GoogleTransWrapper(src_lang="pl", target_lang="en")

    def preprocess_utterance(self, utterance: str) -> str:
        return self.googletrans.translate(utterance)
    
    def preprocess_template(self, template: str) -> str:
        template = super().preprocess_template(template)
        template = self.googletrans.translate(template)
        return template
