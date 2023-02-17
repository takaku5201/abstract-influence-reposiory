import os
import pandas as pd
import numpy as np
import gensim
import sklearn
from sklearn.neural_network import MLPClassifier

root = os.path.dirname(os.path.abspath(__file__))

tarin_file_path = os.path.join(root, "data", "train_data.csv")
test_file_path = os.path.join(root, "data", "test_data.csv")
submission_file_path = os.path.join(root, "data", "submission_data.csv")


# tarin_file = tarin_file_path
# test_file = test_file_path
# submission = submission_file_path


#データの読み込みと前処理
df = pd.read_csv(tarin_file_path, index_col='id')

emb_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

X = []
y = []
for index, data in df.iterrows():
    title = data['title']
    title = title.replace('-', ' ')
    abstract = data['abstract']
    emb_title =  np.mean([emb_model[w.lower()] for w in title.split(' ') if w.lower() in emb_model], axis=0)
    emb_abstract =  np.mean([emb_model[w] for w in abstract.split(' ') if w in emb_model], axis=0)
    if emb_title.ndim == 0:
        emb_title = np.zeros(300)
    X.append(np.concatenate([emb_title, emb_abstract], axis=0))
    y.append(data['y'])

#線形回帰モデルのインスタンス化
model = MLPClassifier()

#予測モデルの作成
model.fit(X, y)

test_df = pd.read_csv(test_file_path, index_col='id')
test_X = []
for index, data in test_df.iterrows():
    title = data['title']
    title = title.replace('-', ' ')
    abstract = data['abstract']
    emb_title =  np.mean([emb_model[w.lower()] for w in title.split(' ') if w.lower() in emb_model], axis=0)
    emb_abstract =  np.mean([emb_model[w] for w in abstract.split(' ') if w in emb_model], axis=0)
    if emb_title.ndim == 0:
        emb_title = np.zeros(300)
    test_X.append(np.concatenate([emb_title, emb_abstract], axis=0))

#テスト結果の出力
test_predicted = model.predict(test_X)
submit_df = pd.DataFrame({'y': test_predicted})
submit_df.reset_index(drop=True)
submit_df.index += 1
submit_df.index.name = 'id'
submit_df.to_csv('submission.csv')