import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def load_model():
    # Load your pre-trained RoBERTa model
    model_path = '/path/to/your/model_state_dict.pth'
    config_path = '/path/to/your/config.json'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, user_input):
    inputs = tokenizer(user_input, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    return "It looks like a non-question." if predicted_label == 0 else "It seems like a question."
