import torch
from transformers import BertTokenizer, BertModel
from transformers.utils import cached_file, WEIGHTS_NAME, CONFIG_NAME

class BERTTextEncoder:
    def __init__(self, model_name="bert-base-uncased", device=None):
        # Set device to CUDA if available, else CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Load the BERT model and move it to the selected device
        try:
            # Check if model files are already cached
            cached_file(model_name, WEIGHTS_NAME)
            cached_file(model_name, CONFIG_NAME)
            local_files_only = True
        except Exception:
            local_files_only = False
        self.model = BertModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)

    def encode(self, text):
        # Tokenize the input text and convert to PyTorch tensors
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
        # Return the embedding of the [CLS] token (first token)
        return outputs.last_hidden_state[:, 0, :]
