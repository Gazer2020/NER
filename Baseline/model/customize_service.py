from vocab import Vocab
import os
import torch
import bilstm_crf
import utils
import random
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model_service.pytorch_model_service import PTServingBaseService
except:
    PTServingBaseService = object

    
def read_data(filepath):
    sentences = []
    sent = ['<START>']
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + ['<END>'])
                sent = ['<START>']
            else:
                sent.append(line[0])
    return sentences


def batch_iter(data, batch_size=32, shuffle=True):
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x), reverse=True)
        yield batch
    
    
class CustomizeService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.code_url = os.path.dirname(os.path.abspath(__file__))
        self.sent_vocab = Vocab.load(os.path.join(self.code_url, "vocab/sent_vocab.json"))
        self.tag_vocab = Vocab.load(os.path.join(self.code_url, "vocab/tag_vocab.json"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"use {self.device}")
        self.model = bilstm_crf.BiLSTMCRF.load(os.path.join(self.code_url, "model.pth"), self.device)
        self.model.eval()

    def _preprocess(self, data):
        preprocessed_data = {}
        for _, v in data.items():
            for file_name, file_content in v.items():
                with open(file_name, "wb") as f:
                    f.write(file_content.read())
                sentences = read_data(file_name)
                sentences = utils.words2indices(sentences, self.sent_vocab)
                preprocessed_data[file_name] = sentences
        return preprocessed_data

    def _inference(self, data):
        res = {"result": []}
        for file_name, test_data in data.items():
            for sentences in batch_iter(test_data, batch_size=int(1), shuffle=False):
                padded_sentences, sent_lengths = utils.pad(sentences, self.sent_vocab[self.sent_vocab.PAD], self.device)
                predicted_tags = self.model.predict(padded_sentences, sent_lengths)
                for sent, pred_tags in zip(sentences, predicted_tags):
                    sent, pred_tags = sent[1: -1], pred_tags[1: -1]
                    for token, pred_tag in zip(sent, pred_tags):
                        res["result"].append(' '.join([self.sent_vocab.id2word(token), self.tag_vocab.id2word(pred_tag)]) + '\n')
                    res["result"].append('\n')
        return res

    def _postprocess(self, data) -> dict:
        logger.info("in postprocess")
        return data
