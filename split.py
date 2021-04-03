import time
from resnet_attention import run
import numpy as np
import gc
import tqdm
import sklearn

X = np.load("X.npy")
Y = np.load("Y.npy")

def split(inc):
    
    x, X, y, Y = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True)

    print(X.shape[0]*X.shape[1]//inc)
    out_X = np.zeros((X.shape[0]*X.shape[1]//inc, X.shape[1]))
    out_Y = np.zeros((X.shape[0]*X.shape[1]//inc, 1))
    counter = 0
    for i in range(X.shape[0]):
        for na in range(1, X.shape[1]-1, inc):
            out_X[counter, 0:na] = X[i,0:na]
            out_Y[counter, :] = Y[i]
            counter += 1

    out_x = np.zeros((x.shape[0]*x.shape[1]//inc, X.shape[1]))
    out_y = np.zeros((x.shape[0]*x.shape[1]//inc, 1))
    counter = 0
    for i in range(x.shape[0]):
        for na in range(1, x.shape[1]-1, inc):
            out_x[counter, 0:na] = x[i,0:na]
            out_y[counter, :] = y[i]
            counter += 1
    scaler = sklearn.preprocessing.MinMaxScaler()

    out_x = np.expand_dims(scaler.fit_transform(out_x), 1)
    out_X = np.expand_dims(scaler.transform(out_X), 1)
    
    return out_X, out_Y, out_x, out_y, scaler

X, Y, x, y, scaler = split(5) # single sample split

# np.save("X_sample.npy", X)
# np.save("Y_sample.npy", Y)
start = time.time()
run(5, X, Y, x, y)
end = time.time() - start
print(end)
