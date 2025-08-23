from transformers import BertForSequenceClassification, BertTokenizer
model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

text='''The US has "passed the peak" on new coronavirus cases, the White House reported. They predict that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.'''
wrong_summary = '''The corona has affected the US'''

input_dict = tokenizer(text, wrong_summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
logits = model(**input_dict).logits
pred = logits.argmax(dim=1)
model.config.id2label[pred.item()] # prints: INCORRECT
