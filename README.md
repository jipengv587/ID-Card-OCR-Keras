
# 1. 项目简介
按照《课工场产品开发合同》（合同编号：）要求，基于Keras深度学习网络实现的身份证信息识别系统。该系统由两部分构成：

* 1.1 文本检测模块：使用Connectionist Text Proposal Network(CTPN)检测图片中文本区域。
* 1.2 文本识别模块：使用DenseNet + Connectionist Temporal Classification(CTC)对文本区域中的文字进行识别。

# 2. 项目部署
将项目文件复制到服务器指定目录，修改根目录下的部署脚本(setup.sh)。请根据实际运行环境，注释掉CPU或者GPU指令。
```
sh setup.sh
```

建议您将根目录路径写入环境变量PYTHONPATH，这可以减少很多引用错误。您可以在您的家目录下的.bashrc或者.bash_profile末尾添加如下内容：

```
export PYTHONPATH=/PATH_TO_YOUR_PROJECT:$PYTHONPATH
```

关闭当前命令行窗口，新开一个窗口后确认环境变量设置已经生效。

```
echo $PYTHONPATH
/PATH_TO_YOUR_PROJECT:/SOME_OTHER_PATH
```

# 3. 项目测试
首先将测试用的身份证图片放入test_images目录后。图片要求：

* 每张图片只能包含一张身份证，卡片横向放置。
* 卡片尽量充满图片。

<img width="400px" src="https://mmbiz.qpic.cn/mmbiz_jpg/OmicicwoEGZZGicv3Z75R4bCOpn98SibxAQLicQUvkubic2800GfqWicuROGZbetWaF4vZpsSvJ5pxVrJE5vibsRarVR5Q/0?wx_fmt=jpeg" />

然后运行下列脚本。

```
python demo.py
```

识别结果会保存到test_result目录。

# 4. 模型训练

模型训练可以分为两部分：文本检测模型(CTPN)训练和文本识别模块(DenseNet+CTC)训练。您可以根据需要选择相应模型进行训练，提升系统整体表现。

## 4.1 文本区域检测模型(CTPN)

我们帮您预训练的模型已经足够精准，建议您在训练之前先测试一下她的表现。若系统提示No module named easydict，请安装easydict模块。

```
pip install easydict
```


测试时，先把测试图片放在ctpn/data/demo目录，然后进入ctpn/ctpn目录直接运行python demo.py。结果文件会保存在ctpn/data/results，文字部分会被标出。

<img width="400px" src="https://mmbiz.qpic.cn/mmbiz_jpg/OmicicwoEGZZGicv3Z75R4bCOpn98SibxAQLmQGV7U7n2WrK2zOp0JicnHvyVNTwHkaeORJufgpgYTckgBSTWFr9nCg/0?wx_fmt=jpeg" />

预训练好的模型数据保存在ctpn/checkpoints目录中。

如果您决定要进行训练，请先下载 [VOCdevkit数据集(1.06G)](https://pan.baidu.com/s/1jVyRiVqWUdFpnjI2HEFmdA) 提取码uaxf。下载并解压缩数据集保存到ctpn/data/VOCdevkit2007目录，然后确认ctpn/ctpn/lib/fast_rcnn/config.py第213行的__C.DATA_DIR正确指向该目录。

```
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
```

下载[VGG_Imagenet预训练模型(553M)](https://pan.baidu.com/s/1jARhwO8bXVgzO_w3P426qw) 提取码ed3u。复制到目录“ctpn/data/pretain_model”。最后，进入目录/ctpn/ctpn运行python train_net.py，出现下列信息表示训练正常。

```
Using config:
{'ANCHOR_SCALES': [16],
 'DATA_DIR': '/Path_to_your_project_root/ctpn/data',
 'DEDUP_BOXES': 0.0625,
 'EPS': 1e-14,
 'EXP_DIR': 'ctpn_end2end',
 'GPU_ID': 0,
 'IS_EXTRAPOLATING': True,
 'IS_MULTISCALE': False,
 'IS_RPN': True,
 'LOG_DIR': 'ctpn',
 ...
 ...
 assign pretrain model weights to conv2_1
 assign pretrain model biases to conv2_1
 iter: 0 / 50000, total loss: 2.2971, model loss: 1.6194, rpn_loss_cls: 0.6704, rpn_loss_box: 0.9490, lr: 0.000010
 speed: 22.306s / iter
```

训练生成的数据会保存在ctpn/output目录。

## 4.2 DenseNet+CTC训练

决定训练之前，建议您测试一下当前系统的表现。我们提供的预训练模型已经十分精准(accuracy:90%)，您可以按照“3.项目测试”中的方法进行测试。如果您决定要训练，请先下载数据集

* [中文字符图片集(8.6G)](https://pan.baidu.com/s/1AQGy7uTVbZaDbJ80kvCGsw) 提取码: a8i8
* [训练数据标签(205M)](https://pan.baidu.com/s/1O2_U1pa4viwEm1cxBEPgqA) 提取码: cxf7 
* [测试数据标签(2M)](https://pan.baidu.com/s/1fOHfc2mJcHZx8Y-xJQl8vg) 提取码: xdnf

下载并解压缩图片集后放到train/images目录下，数据标签文件放到train目录下即可。进入目录train开始训练

```
cd train
python train.py
```

# 5. 效果展示

<img width="800px" src="https://mmbiz.qpic.cn/mmbiz_jpg/OmicicwoEGZZEHOCZbynj5WMnPcsf4kDrqNwy3VaVxDwIKicUxVXP6LicWAUXaYM9LGibNzOKfzbQx6VrvRWducc5JQ/0?wx_fmt=jpeg"/>

