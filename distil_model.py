import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

def distil_model(texts):
    num_labels = 4
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict('distil_model.pt')
    model.eval()
    model.to(device)
    
    predictions = []
    for text in texts:
        test_encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        input_ids = test_encoding['input_ids'].to(device)
        attention_mask = test_encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        predictions.append(preds.item())
    
    return predictions