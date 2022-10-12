import mymodule as mm
import numpy as np
import gzip
import struct
from scipy.special import softmax
import numdifftools as nd

def softmax_loss(Z, y):
	log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))
	for i in range(y.shape[0]):
			log_sum_exp[i] -= Z[i, y[i]]
	l = np.mean(log_sum_exp)
	return l

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=50):
    for i in range(0, y.shape[0], batch):
      x = X[i:i+batch]
      y_b = y[i:i+batch]
      Z = np.transpose(np.exp(np.dot(x, theta)).T/(np.sum(np.exp(np.dot(x, theta)), axis=1)))      
      I = np.zeros((Z.shape[0], theta.shape[1]))      
      I[range(batch), y_b] = 1
      theta -= (lr/batch)*(np.dot(x.T, (Z-I)))   
    return theta

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
  magic_num, items = struct.unpack_from('>ii', f.read(8))
  rows, columns = struct.unpack('>ii', f.read(8))
  byte_stream = f.read()
  images = np.frombuffer(byte_stream, dtype='uint8')
  images = images.reshape((items, rows*columns)).astype(np.float32)
  images = images/np.max(images)


with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
  magic_num, items = struct.unpack_from('>ii', f.read(8))
  byte_stream = f.read()
  labels = np.frombuffer(byte_stream, dtype='uint8')
  labels = labels.reshape(items).astype(dtype=np.uint8)


ex_size = 1000
# X = np.random.randn(ex_size,728).astype(np.float32)
# y = np.random.randint(10, size=(ex_size,)).astype(np.uint8)
# Theta_ = np.zeros((728,10), dtype=np.float32)
# Theta = np.zeros((728,10), dtype=np.float32)
np.random.seed(0)
X = images
y = labels
Theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)
Theta_ = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)

# # mm.softmax_regression_epoch_cpp(X, y, Theta, lr=1, batch=50)


# X = np.random.randn(500,5).astype(np.float32)
# y = np.random.randint(3, size=(500,)).astype(np.uint8)
# Theta_ = np.zeros((5,3), dtype=np.float32)
# Theta = np.zeros((5,3), dtype=np.float32)
theta = softmax_regression_epoch(X[:100], y[:100], Theta_, lr=0.2, batch=100)
mm.softmax_regression_epoch_cpp(X[:100], y[:100], Theta, lr=0.2, batch=100)
# dTheta = -nd.Gradient(lambda Th : softmax_loss(X@Th.reshape(X.shape[1],y.max()+1),y))(Theta)



# print(dTheta.reshape(X.shape[1], y.max()+1))
print(np.linalg.norm(theta))


# print(np.linalg.norm(theta), np.linalg.norm(dTheta))