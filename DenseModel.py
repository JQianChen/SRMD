from keras.layers import Conv2D, Input, concatenate, Lambda, add, merge
from keras.models import Model
import tensorflow as tf
from keras.utils.vis_utils import plot_model  # 可视化


# 该模型为粗特征提取部分为多尺度提取，改进的八个DenseBlock(每个denseBlock都有多尺度提取预处理)，特征稠密融合，降维，子像素卷积+卷积超分重建，
# 在每次卷积前有一个多尺度卷积
# 子像素卷积 Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)  # tensorflow中的子像素卷积

    return Lambda(subpixel, output_shape=subpixel_shape)


# 计算psnr
def psnr(y_true, y_pred):
    psnr_cal = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr_cal


# 多尺度卷积,4倍filters,含有1x1卷积
def multiConv(block_input, activation='relu', filters=16):
    initializer = 'he_normal'
    # initializer = 'glorot_normal'
    ######################
    model_scale9 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
                          kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
    model_scale9 = Conv2D(filters=filters, kernel_size=(9, 9), padding='same', activation=activation,
                          kernel_initializer=initializer, )(model_scale9)

    #####################
    model_scale7 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
                          kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度

    model_scale7 = Conv2D(filters=filters, kernel_size=(7, 7), padding='same', activation=activation,
                          kernel_initializer=initializer, )(model_scale7)

    ####################
    model_scale5 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
                          kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
    model_scale5 = Conv2D(filters=filters, kernel_size=(5, 5), padding='same', activation=activation,
                          kernel_initializer=initializer, )(model_scale5)

    ####################
    model_scale3 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
                          kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
    model_scale3 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                          kernel_initializer=initializer, )(model_scale3)

    model_out = concatenate(axis=-1, inputs=[model_scale3, model_scale5, model_scale7, model_scale9])
    return model_out


# 输出chanel数量是filters的八倍
def dense_block(block_input, activation='relu', filters=16):
    initializer = 'he_normal'
    # initializer = 'glorot_normal'
    # 多尺度融合降维
    model_0 = multiConv(block_input, filters=4)  # 输入16
    ####DenseNet
    model_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                     kernel_initializer=initializer, )(model_0)

    model_2_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer, )(model_1)
    model_2 = concatenate(axis=-1, inputs=[model_1, model_2_1])

    model_3_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer, )(model_2)
    model_3 = concatenate(axis=-1, inputs=[model_2, model_3_1])

    model_4_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer)(model_3)
    model_4 = concatenate(axis=-1, inputs=[model_3, model_4_1])

    model_5_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer)(model_4)
    model_5 = concatenate(axis=-1, inputs=[model_4, model_5_1])

    model_6_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer)(model_5)
    model_6 = concatenate(axis=-1, inputs=[model_5, model_6_1])

    model_7_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer)(model_6)
    model_7 = concatenate(axis=-1, inputs=[model_6, model_7_1])

    model_8_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
                       kernel_initializer=initializer)(model_7)
    model_8 = concatenate(axis=-1, inputs=[model_7, model_8_1])
    return model_8


def DNSR(h=24, w=24, scale=4):
    activation = 'relu'  # 'tanh'
    input = Input([h, w, 1])
    feature = multiConv(input, filters=16)  # 输出64，底层特征
    dense_block_1 = dense_block(feature, activation=activation)
    dense_block_2 = dense_block(dense_block_1, activation=activation)
    dense_block_2 = concatenate(inputs=[dense_block_1, dense_block_2])  ###
    dense_block_3 = dense_block(dense_block_2, activation=activation)
    dense_block_3 = concatenate(inputs=[dense_block_2, dense_block_3])  ###
    dense_block_4 = dense_block(dense_block_3, activation=activation)
    dense_block_4 = concatenate(inputs=[dense_block_3, dense_block_4])  ###
    dense_block_5 = dense_block(dense_block_4, activation=activation)
    dense_block_5 = concatenate(inputs=[dense_block_4, dense_block_5])  ###
    dense_block_6 = dense_block(dense_block_5, activation=activation)
    dense_block_6 = concatenate(inputs=[dense_block_5, dense_block_6])  ###
    dense_block_7 = dense_block(dense_block_6, activation=activation)
    dense_block_7 = concatenate(inputs=[dense_block_6, dense_block_7])  ###
    dense_block_8 = dense_block(dense_block_7, activation=activation)
    dense_block_8 = concatenate(inputs=[dense_block_7, dense_block_8])
    # ####
    out = Conv2D(filters=scale * scale, kernel_size=(1, 1), padding='same')(dense_block_8)  # 特征融合
    out = SubpixelConv2D(scale)(out)  # 上采样
    out = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(out)  # reconstruction
    ####
    DenseNSR = Model(inputs=[input], outputs=[out])
    return DenseNSR


if __name__ == "__main__":
    model, modelName = DNSR(), 'DenseNetSR'
    model.summary()
    plot_model(model, to_file="./data/{}.png".format(modelName), show_layer_names=False, show_shapes=True)
