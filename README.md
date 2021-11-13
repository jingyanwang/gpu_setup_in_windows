# gpu_setup_in_windows

1. check gpu 

<img src="471748e7-7b4d-45c7-bb31-985e0128e4af.jfif" width="350" title="hardware">


2. install cuda version

<img src="WeChat Screenshot_20211113150350.png" width="350" title="hardware">

3. download and install cuda from 

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

<img src="WeChat Screenshot_20211113150657.png" width="400" title="hardware">

when install cuda, remember to choose the right device as you checked in step 1

4. check your version of tensorflow

<img src="WeChat Screenshot_20211113150929.png" width="250" title="hardware">

and find the corresponding version of cudnn from https://www.tensorflow.org/install/source_windows#gpu

<img src="WeChat Screenshot_20211113151124.png" width="400" title="hardware">


5. download cudnn package of the correct version from https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-111 

<img src="WeChat Screenshot_20211113151342.png" width="400" title="hardware">


6. go to folder 

```
cudnn-11.2-windows-x64-v8.1.0.77\cuda
```

and copy all files and paste to 

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5
```

<img src="WeChat Screenshot_20211113151527.png" width="400" title="hardware">


7. run python code to detect the GPU device

```python
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos])
```

<img src="WeChat Screenshot_20211113151801.png" width="400" title="hardware">

8. run dl training program to check if the GPU is used

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review


# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()


(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val))
```

<img src="WeChat Screenshot_20211113152138.png" width="800" title="hardware">
