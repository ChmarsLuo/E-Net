from  data import  *
from keras.models import load_model
import tensorflow as tf
import itertools
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from tqdm import tqdm
from keras.utils import get_custom_objects

##########################################损失函数#######################################################################
# ============================ Jaccard/IoU score ============================
SMOOTH = 1.


# ============================ Jaccard/IoU score ============================
def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数：
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison),
        if ``None`` prediction prediction will not be round
    返回：
        IoU/Jaccard score in range [0, 1]
    '''
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = tf.reduce_mean(iou, axis=0)

    # weighted mean per class
    iou = tf.reduce_mean(iou * class_weights)

    return iou


# 计算IOU得分
def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    def score(gt, pr):
        return iou_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, threshold=threshold)

    return score


jaccard_score = iou_score
get_jaccard_score = get_iou_score

get_custom_objects().update({'iou_score': iou_score,
                             'jaccard_score': jaccard_score})


# ============================== F/Dice - score ==============================

def f_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):
    # F_score（Dice系数）可以解释为精确度和召回率的加权平均值，
    # 其中F-score在1时达到其最佳值，在0时达到最差分数。
    # 精确率和召回率对F1-score的相对影响是一样的，公式表示为：
    # $F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
    #    {\beta^2 \cdot precision + recall}$
    # 公式还有另外一种表达形式：
    # $F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}$
    # 其中 TP表示ture positive
    # FP表示fasle positive
    # FN表示false negtive
    # 参数：
    #   gt: ground truth 4D keras tensor (B, H, W, C)
    #    pr: prediction 4D keras tensor (B, H, W, C)
    #    class_weights: 1. or list of class weights, len(weights) = C
    #    beta: f-score coefficient
    #    smooth: value to avoid division by zero
    #    per_image: if ``True``, metric is calculated as mean over images in batch (B),
    #        else over whole batch
    #    threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round
    # 返回：
    # [0, 1]区间内的F-score

    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    fn = tf.reduce_sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = tf.reduce_mean(score, axis=0)

    # weighted mean per class
    score = tf.reduce_mean(score * class_weights)

    return score


def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round
    返回:
        ``callable``: F-score
    '''

    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image,
                       threshold=threshold)

    return score


f1_score = get_f_score(beta=1)
f2_score = get_f_score(beta=2)
dice_score = f1_score

# Update custom objects
get_custom_objects().update({
    'f1_score': f1_score,
    'f2_score': f2_score,
    'dice_score': dice_score,
})

########################################################################################################################
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)

from keras import backend as K
'''
Compatible with tensorflow backend
'''
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

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


########################################################################################################################
im_width = 256
im_height = 256
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + 'images\\' + id_, color_mode="grayscale")
        # img = load_img(path + 'images\\' + id_, color_mode="rgb")
        x_img = img_to_array(img)
        x_img = resize(x_img, (256, 256,3), mode='constant', preserve_range=True)

        fname, extension = os.path.splitext(id_)
        if extension != '.png':
            mask_id_ = fname + '.png'
        else:
            mask_id_ = id_
        # Load masks
        if train:
            mask = img_to_array(load_img(path + '\\masks\\' + mask_id_, color_mode="grayscale"))
            mask = resize(mask, (256, 256, 1), mode='constant', preserve_range=True)

        # Save images
        # print(X.shape)
        # print(x_img.shape)
        # X[n, ..., 0] = x_img.squeeze() / 255
        # print(X.shape)
        # print(x_img.shape)
        # X[n, ...] = x_img.squeeze()/255.0
        X[n, ...] = x_img / 255.0
        # print(X.shape)
        # print(x_img.shape)
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X

path = 'C:\\Users\\Charm Luo\\Desktop\\my-data\\dense_seg\\crop512\\val\\'
X = get_data(path, train=False)
#print(X)
class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.

    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.

        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """

        pred = []
        for x_i in X:
            # x_i = cv2.imread(x_i,1)
            # print(x_i.shape)
            # x_i = trans.resize(x_i, (256, 256, 3))
            # x_i = np.squeeze(x_i)
            # x_i = np.expand_dims(x_i,0)
            # # print(x_i.shape)
            # p0 = self.model.predict(x_i)
            # p1 = self.model.predict(np.fliplr(x_i))
            p0 = self.model.predict(self._expand(np.fliplr(x_i[:, :, 0])))
            p1 = self.model.predict(self._expand(np.fliplr(x_i[:, :, 0])))
            #p2 = self.model.predict(self._expand(np.flipud(x_i[:, :, 0])))
            #p3 = self.model.predict(self._expand(np.fliplr(np.flipud(x_i[:, :, 0]))))
            # print(p0.shape)
            # print(np.max(p0), np.min(p0))
            # print(p1.shape)
            p = (p0 +
                 self._expand(np.fliplr(p1[0][:, :, 0])) #+
             #    self._expand(np.flipud(p2[0][:, :, 0])) +
             #   self._expand(np.fliplr(np.flipud(p3[0][:, :, 0])))
                 ) / 2
            pred.append(p)
        return np.array(pred)

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)
########################################################################################################################

if __name__ == '__main__':
    test_img_path = './build/test/images/'
    save_path = './output/build-1-result/'
    names = os.listdir(test_img_path)
    test_number = len(names)
    batch_size = 4
    # model = load_model('.\\logs\\ccd4.h5',custom_objects={'weighted_binary_crossentropy':wb_loss.weighted_binary_crossentropy})
    # model = load_model('C:\\Users\\Charm Luo\\Desktop\\my-data\\dense_seg\\dense_seg2\\logs\\ccd3.h5')#, custom_objects={'accuracy': accuracy})
    # model = load_model('.\\logs\\ccd3.h5',custom_objects={'dice_coef_loss':dice_coef_loss})
    model = load_model('./output/build-1-weight/build_012.h5',
                       custom_objects= {'iou_score':iou_score, 'dice_score':dice_score, 'f1_score':f1_score, 'f2_score':f2_score} )
    testGene = testGenerator(test_img_path)
    # tta_model = TTA_ModelWrapper(model)
    # result_tta = tta_model.predict(X)
    results = model.predict_generator(testGene, steps = test_number, verbose=1)
    saveResult(save_path, results, names)