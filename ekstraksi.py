from transformers import BertTokenizer

if __name__ == '__main__':

    
    tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    data = tokenizer("saya sedang makan siang")

    print(data)