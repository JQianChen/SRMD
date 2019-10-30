import glob
from random import shuffle  # 打乱列表
import numpy as np

from DenseModel import psnr
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import threading
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras.backend as K
import skimage.measure as judge
from PIL import Image
import DenseModel as model
# 基于fit_generate()训练，即边加载图像边训练模型，能够处理大数据集的训练


import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import scipy.io as sio

################################################   keras使用GPU
def Gpuconfig():
    # 指定第一块GPU可用
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU的第二种方法

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 1  # 定量
    config.gpu_options.allow_growth = True  # 按需
    set_session(tf.Session(config=config))
################################################

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


# 生成图像路径list
def load_pic(_label_path) -> list:
    pic_path_list = []
    images = glob.glob(_label_path)
    for item in images:
        pic_path_list.append(item)
    shuffle(pic_path_list)
    return pic_path_list


# 从图像list中选取patch_list
def batch_path_list(pic_path_list, offset, batch_size) -> list:
    _batch_path_list = pic_path_list[offset:offset + batch_size]
    return _batch_path_list


# 输入一个图像路径列表，加载列表中所有图像，每张图像抽样patch,缩小2倍作为输入input_patch,源图像作为label图像
def batch_patch(_batch_path):
    global scale
    input_patch = []
    label_patch = []
    for item in _batch_path:
        label_image = Image.open(item)
        w, h = label_image.size
        # assert isinstance(label_image, Image.Image)
        input_image = label_image.resize((w // scale, h // scale), resample=Image.BICUBIC)

        label_image = np.array(label_image)
        input_image = np.array(input_image)

        label_image = label_image[:, :, np.newaxis]
        label_patch.append(label_image)  # 源图像作为label
        input_image = input_image[:, :, np.newaxis]
        input_patch.append(input_image)
    return input_patch, label_patch


def batch_generate(_pic_path_list, batch_size):
    while True:  # 生成器要一直运行，即无限循环，但不是真的在运行,每次调用运行到yield就会返回，否则会抛出StopIteration异常
        for step in range(len(_pic_path_list) // batch_size):
            offset = step * batch_size
            _batch_path_list = batch_path_list(_pic_path_list, offset, batch_size)
            input_patch, label_patch = batch_patch(_batch_path_list)
            input_patch = np.array(input_patch, np.float32) / 255.0  # 归一化到0-1
            label_patch = np.array(label_patch, np.float32) / 255.0
            # print(input_patch.shape, label_patch.shape)
            yield (input_patch, label_patch)


def train():
    train_path = "/home/laglangyue/AApython/Data/crop/DIV2K_96_1/train/*"
    test_path = "/home/laglangyue/AApython/Data/crop/DIV2K_96_1/test/*"

    global scale
    scale = 4
    batch_size = 32
    epoch = 200
    # 生成路径列表
    train_list = load_pic(train_path)
    test_list = load_pic(test_path)
    # 定义模型并训练
    Gpuconfig()
    DNSR = model.DNSR(scale=scale)
    adam = Adam(lr=0.001)

    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 50 == 0 and epoch != 0:
            lr = K.get_value(DNSR.optimizer.lr)
            K.set_value(DNSR.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(DNSR.optimizer.lr)

    DNSR.compile(optimizer=adam, loss='mean_squared_error', metrics=[psnr])
    checkpoint = ModelCheckpoint("./data/h5/DenseNetSR.h5", monitor='loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler)  ##学习率衰减,20次衰减
    callbacks_list = [checkpoint, reduce_lr]
    history = DNSR.fit_generator(batch_generate(train_list, batch_size),
                                 steps_per_epoch=len(train_list) // batch_size,
                                 validation_data=batch_generate(test_list, batch_size),
                                 validation_steps=len(test_list) // batch_size,
                                 epochs=epoch, workers=8, verbose=2,
                                 callbacks=callbacks_list)

    # workers: 整数。使用的最大进程数量，如果使用基于进程的多线程。 如未指定，workers 将默认为 1。如果为 0，将在主线程上执行生成器。
    # 绘制训练 & 验证的损失值
    plt.plot(history.history['psnr'])
    plt.plot(history.history['val_psnr'])
    sio.savemat("./SRMDenseNet.mat",
                {'loss': history.history['loss'],
                 'val_loss': history.history['val_psnr'],
                 'psnr': history.history['psnr'],
                 'val_psnr': history.history['val_psnr']})
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def predict(path, result_path, scale):
    '''
    :param path:输入图像的文件夹
    :param result_path: 经过超分模型输出图像的文件夹
    :param scale: 超分倍率
    :return: 0
    '''
    files = glob.glob(path)
    ###创建结果文本文件
    result_txt_path = result_path + "result_{}.txt".format(scale)
    result_txt = open(result_txt_path, "w+")
    result_txt.truncate()  # 清空文件内容
    psnr_mean = []
    ###
    for file in files:
        img = Image.open(file)
        assert isinstance(img, Image.Image)
        img = img.convert("YCbCr")
        img, _, _ = img.split()
        ################################################# 去除不能整除部分
        w, h = img.size
        ##########
        img = img.crop([0, 0, w - w % scale, h - h % scale])  # 保证整除
        original = np.array(img)
        ##########
        #################################################
        img_name = result_path + file.split("/")[-1].split(".")[0] + ".bmp"
        #####下采样
        img = img.resize((w // scale, h // scale), Image.BICUBIC)
        img = np.array(img)
        #####
        img = img / 255.  # 归一化到0-1
        img = img[np.newaxis, :, :, np.newaxis]
        Gpuconfig()
        Model = model.DNSR(h=img.shape[1], w=img.shape[2], scale=scale)
        Model.load_weights("./data/h5/DenseNetSR.h5")  #######
        pre = Model.predict(img, batch_size=1)
        print(img_name)
        pre = np.clip(pre, 0, 1) * 255
        pre = pre.astype(np.uint8)
        ####
        pre = np.squeeze(pre, 0)  # 去除第0维
        pre = np.squeeze(pre, -1)
        pre_image = Image.fromarray(pre)  # numpy转Image
        pre_image.save(img_name)
        ####

        psnr = judge.compare_psnr(original, pre)
        # print(psnr)
        # 写入 Y分量的 psnr
        psnr_mean.append(psnr)
        result_txt.write("{}".format(psnr) + "  " + file + "    " + img_name + "\n")
    ###写入PSNR均值到文本文件
    result_txt.write("{}".format(np.mean(psnr_mean)) + "\n")
    result_txt.write("-----------------------------------------")
    result_txt.close()


def predictSet():
    # data = "Set5"
    # path = "/home/laglangyue/AApython/Data/SetData/Test/{}/*.bmp".format(data)
    # data = "Set14"
    # path = "/home/laglangyue/AApython/Data/SetData/Test/{}/*.bmp".format(data)
    # data = "B100"
    # path = "/home/laglangyue/AApython/Data/BSDS300/valid/*"
    data = "U100"
    path = "/home/laglangyue/AApython/Data/Urban100/*" #路径
    global scale
    scale = 4
    result_path = "./result/{}_X4/".format(data)
    predict(path, result_path, scale)

if __name__ == "__main__":
    #train()
    predictSet()
