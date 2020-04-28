# 3-2,中阶API示范

### 二， DNN二分类模型

**1，准备数据**

**2, 定义模型**

```python

```

```python
class DNNModel(tf.Module):
    def __init__(self,name = None):
        super(DNNModel, self).__init__(name=name)
        self.dense1 = layers.Dense(4,activation = "relu") 
        self.dense2 = layers.Dense(8,activation = "relu")
        self.dense3 = layers.Dense(1,activation = "sigmoid")

     
    # 正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def __call__(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y
    
model = DNNModel()
model.loss_func = losses.binary_crossentropy
model.metric_func = metrics.binary_accuracy
model.optimizer = optimizers.Adam(learning_rate=0.001)

```

```python
# 测试模型结构
(features,labels) = next(ds.as_numpy_iterator())

predictions = model(features)

loss = model.loss_func(tf.reshape(labels,[-1]),tf.reshape(predictions,[-1]))
metric = model.metric_func(tf.reshape(labels,[-1]),tf.reshape(predictions,[-1]))

tf.print("init loss:",loss)
tf.print("init metric",metric)

```

```
init loss: 1.13653195
init metric 0.5
```

```python

```

**3，训练模型**

```python
#使用autograph机制转换成静态图加速

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    grads = tape.gradient(loss,model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads,model.trainable_variables))
    
    metric = model.metric_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    
    return loss,metric

# 测试train_step效果
features,labels = next(ds.as_numpy_iterator())
train_step(model,features,labels)
```

```
(<tf.Tensor: shape=(), dtype=float32, numpy=1.2033114>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.47>)
```

```python

```

```python
def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss, metric = tf.constant(0.0),tf.constant(0.0)
        for features, labels in ds:
            loss,metric = train_step(model,features,labels)
        if epoch%10==0:
            printbar()
            tf.print("epoch =",epoch,"loss = ",loss, "accuracy = ",metric)
train_model(model,epochs = 60)

```

```
================================================================================17:07:36
epoch = 10 loss =  0.556449413 accuracy =  0.79
================================================================================17:07:38
epoch = 20 loss =  0.439187407 accuracy =  0.86
================================================================================17:07:40
epoch = 30 loss =  0.259921253 accuracy =  0.95
================================================================================17:07:42
epoch = 40 loss =  0.244920313 accuracy =  0.9
================================================================================17:07:43
epoch = 50 loss =  0.19839409 accuracy =  0.92
================================================================================17:07:45
epoch = 60 loss =  0.126151696 accuracy =  0.95
```

```python

```

