from model import pre_train, predict_lstm_gen
from CN.stock import Stock
s = Stock('002230')
data = s.get_data()['close']
input_size,target_len = 100, 1
base = data.max()
last_idx = data.idxmax()
lstm_model = pre_train(data,base,input_size,target_len)
pred = predict_lstm_gen(data.iloc[last_idx+1-input_size:]/base,lstm_model,input_size,target_len)*base

print(pred[-target_len])