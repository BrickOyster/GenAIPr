from transformers import BertModel, BertTokenizer
import spacy
from openie import StanfordOpenIE
import torch

class BERTTextEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")
        self.openie = StanfordOpenIE()
        
    def get_sentence_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    def extract_triples(self, text):
        doc = self.nlp(text)
        triples = self.openie.annotate(text)
        return triples
    
if __name__ == "__main__":
    # Example usage
    text = "A bed is near the window. A lamp is on the nightstand."
    encoder = BERTTextEncoder()
    embedding = encoder.get_sentence_embedding(text)
    print("Sentence Embedding:", embedding)
    triples = encoder.extract_triples(text)
    print("Extracted Triples:")
    for triple in triples:
        print(f"Subject: {triple['subject']}, Relation: {triple['relation']}, Object: {triple['object']}")
    pass