import transformers

MAX_LEN = 512
TRAIN_BATH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS =10
ACCUMULATION = 2
BERT_PATH = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
MODEL_PATH = "model.bin"
TRAINING_FILE = "..\\input\\imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = True)
