## 使用卷积神经网络对omniglot进行分类

### 数据处理

#### 数据集简介

分类的数据集为omniglot数据集。数据集包含来自 50 个不同字母的 1623 个不同手写字符。每一个字符都是由 20个不同的人通过亚马逊的 Mechanical Turk 在线绘制的。也就是说作为分类任务来说，一共有1623个类别，每个类别有20个数据，经过1：9分割后有18个作为trian数据，2个作为eval数据。

#### 加载数据

```python

def load_images(path,n=0):
    X = []
    Y=[]
    i=-1
    
    for back in os.listdir(path):
        back_path = os.path.join(path,back)
        for language in os.listdir(back_path):
            alphabet_path = os.path.join(back_path,language)
            for letter in os.listdir(alphabet_path):
                category_images = []
                i=i+1
                letter_path = os.path.join(alphabet_path,letter)

                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path,filename)
                    image = imread(image_path)
                    image = imresize(image,(28,28))
                    image = image/255
                    image = 1-image
                    Y.append(i)
                    X.append(image)
    return X,Y
```

得到X，Y。X每个数据为28X28的矩阵，共有32460条。Y为一个代表种类的数，对应有3240条。

只有对XY进行分割，将其中十分之一用于评估。

```python
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1,random_state = 0)
```

之后将Y转化为独热向量方便计算交叉熵。

```python


number_of_classes = 1623
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
```

### 模型建立和实验
#### 建立模型
```python

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3 )))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1623))

model.add(Activation('softmax'))
```

![2020-05-03 10-17-25 的屏幕截图](/home/zhangao/图片/Screenshots/2020-05-03 10-17-25 的屏幕截图.png)

编译模型：优化器选择Adam，以acc为指标。

```python
model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
```

#### 训练模型

定义generator，并调用fit_generator训练。


```python
gen = ImageDataGenerator()
test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=32)
test_generator = test_gen.flow(X_test, Y_test, batch_size=32)
model.fit_generator(train_generator, epochs=100, validation_data=test_generator)
```

训练得到的acc和loss如下。

![acc](/home/zhangao/acc.png)
![loss](/home/zhangao/omn.png)

#### 评估模型

```python
score = model.evaluate(X_test, Y_test)
print()
print('Test loss: ', score[0])
print('Test Accuracy', score[1])
```

```
Test loss:  0.584456355664
Test Accuracy 0.854432345644
```