import py_vncorenlp
import re
import numpy as np
from gensim.models import KeyedVectors
import torch
from torch import nn

matrix_size = (300,300)

class GRUmodel(nn.Module):
  def __init__(self):
    super().__init__()

    self.rnn = nn.GRU(input_size=matrix_size[1], hidden_size=441,
                      num_layers=3, batch_first=True, bidirectional=False)

    self.fc = nn.LazyLinear(out_features=1)

  # it output [0,1,2,....,seq_length - 1]
  # just take the last array element in case of classification or anything like that
  def forward(self, X, state=None):
    rnn_outputs, _ = self.rnn(X, state)

    return self.fc(rnn_outputs[:, -1, :])

  def feature_extract(self, X, state=None):
    rnn_outputs, _ = self.rnn(X, state)
    return rnn_outputs[:, -1, :]

PATH='/home/nhatdm2k4/Pictures/Data_mining/demo/model'
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=PATH)





def convert2segment(sentence):
    output = rdrsegmenter.word_segment(sentence)
    output = (' '.join(output))
    output = re.sub('[^A-Za-z_À-Ỹà-ỹĂ-Ẽă-ẽẮ-Ỷắ-ỷẰ-Ỹằ-ỹẤ-Ỵấ-ỵẦ-Ỵầ-ỵẢ-Ỷả-ỷẲ-Ỹẳ-ỹẨ-Ỵẩ-ỵẠ-Ỵạ-ỵƠ-Ỹơ-ỹỚ-Ỷớ-ỷỜ-Ỹờ-ỹỞ-Ỷở-ỷỠ-Ỹỡ-ỹỢ-Ỵợ-ỵ]',
        ' ', output).lower()
    return output

def embedding_300(text):
  model = KeyedVectors.load('/home/nhatdm2k4/Pictures/Data_mining/demo/model/word2vec')
  text = text.split()
  n_size = len(text)
  n_feature = 300
  embed = []
  if n_size <= n_feature:
    for x in text:
      cur = np.zeros(n_feature)
      if model.has_index_for(x):
        cur = model.get_vector(x)
      embed.append(cur)
    for _ in range(n_feature-n_size):
      embed.append(model.get_vector('#'))
  else:
    for x in text[:(n_feature-100)//2]:
      cur = np.zeros(n_feature)
      if model.has_index_for(x):
        cur = model.get_vector(x)
      embed.append(cur)
    s = (n_feature-100)//2 + (n_size-n_feature)//2
    for x in text[s:s+100]:
      cur = np.zeros(n_feature)
      if model.has_index_for(x):
        cur = model.get_vector(x)
      embed.append(cur)
    for x in text[-(n_feature-100)//2:]:
      cur = np.zeros(n_feature)
      if model.has_index_for(x):
        cur = model.get_vector(x)
      embed.append(cur)
  # print(len(embed), arr.shape)
  return np.array(embed)

def inference(input):
  input.resize(1, 300, 300)
  model = GRUmodel()
  model.load_state_dict(torch.load('/home/nhatdm2k4/Pictures/Data_mining/demo/model/best_GRUmodel.pth', map_location=torch.device('cpu')))
  tensor_data = torch.Tensor(input)
  model.eval()

  with torch.inference_mode():
    output = model(tensor_data).squeeze()
    # dua ve dang [0, 1]
    proba = torch.sigmoid(output)
    proba = np.array(proba.to('cpu'))
    # dua ve 0/1
    # pred = torch.round(proba)
  return proba

if __name__ == '__main__':
  input = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
  input = convert2segment(input)
  input = embedding_300(input)
  print(inference(input))