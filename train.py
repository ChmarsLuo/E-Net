from data import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard,EarlyStopping, CSVLogger
from model import U_NET
from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import itertools
import os


#####################################################损失函数############################################################
def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
########################################################################################################################
class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)


class WeightedCategoricalCrossEntropy(object):

    def __init__(self, weights):
        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'w_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.w_categorical_crossentropy(y_true, y_pred)

    def w_categorical_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
########################################################################################################################


#################################################### prepare model######################################################
model = U_NET(2) #decoder_block_type='transpose')
model.summary()
model.compile(optimizer=Adam(lr=1.0e-3), loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer=Adam(lr=1.0e-3), loss='binary_crossentropy', metrics=[my_iou_metric])
# model.compile(optimizer=Adam(lr=1.0e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=1.0e-3), loss='categorical_crossentropy', metrics=[my_iou_metric])
# model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[my_iou_metric])
# model.compile(optimizer=Adam(lr=1.0e-3), loss= bce_dice_loss, metrics=[my_iou_metric])
# # wb_loss = WeightedBinaryCrossEntropy(0.0946) #调用类
# # model.compile(optimizer=Adam(lr=1e-3), loss=wb_loss.weighted_binary_crossentropy, metrics=['accuracy'])
########################################################################################################################


###########################################################学习率调整####################################################
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 1 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.95)
        print("lr changed to {}".format(lr * 0.95))
    return K.get_value(model.optimizer.lr)
####################################################train model#########################################################
if __name__ == '__main__':
    batch_size = 4
    img_size = 256
    epochs = 100
    train_im_path,train_mask_path = './build/train/images/','./build/train/labels/'
    val_im_path,val_mask_path = './build/val/images/','./build/val/labels/'
    train_set = os.listdir(train_im_path)
    val_set = os.listdir(val_im_path)
    train_number = len(train_set)
    val_number = len(val_set)

    train_root = './build/train/'
    val_root = './build/val/'
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    training_generator = trainGenerator(batch_size,train_root,'images','labels',data_gen_args,save_to_dir = None)
    validation_generator = trainGenerator(batch_size,val_root,'images','labels',data_gen_args,save_to_dir = None)


    model_path ="./output/build-1-weight/"
    model_name = 'build_{epoch:03d}.h5'
    model_file = os.path.join(model_path, model_name)
    model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    lr_reducer = LearningRateScheduler(scheduler)
    # model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=False)
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.5625), cooldown=0, patience=5, min_lr=0.5e-6)
    callable = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
                model_checkpoint,
                lr_reducer,
                CSVLogger(filename='./output/build-1-log/log.csv', append=False),  # CSVLoggerb保存训练结果   好好用
                TensorBoard(log_dir='./output/build-1-log/')]

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=train_number//batch_size,
                        validation_steps=val_number//batch_size,
                        use_multiprocessing=False,
                        epochs=epochs,verbose=1,
                        initial_epoch = 3,
                        callbacks=callable)