from myconv_keras import build_mycov, build_myconv01, build_myconv03, build_mycov11, build_mycov12
from image_util import ImageDataProvider
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
import os
from helper_functions import *


log_path = "./trained/log/"


generator = ImageDataProvider("../data/bowl/train/*.tif")
verifyGenerator = ImageDataProvider("../data/bowl/verify/*.tif")
x_train, y_train = generator(374)
x_valid, y_valid = verifyGenerator(152)


print(">> Train data: {} | {} ~ {}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print(">> Train mask: {} | {} ~ {}\n".format(y_train.shape, np.min(y_train), np.max(y_train)))
print(">> Valid data: {} | {} ~ {}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print(">> Valid mask: {} | {} ~ {}\n".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

builder = 'unet++'

if builder == 'myconv':
    model_name = "myconv"
    model = build_mycov(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./trained/model/myconv/"

if builder == 'myconv11':
    model_name = "myconv11"
    model = build_mycov11(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./trained/model/myconv11/"

if builder == 'myconv12':
    model_name = "myconv12"
    model = build_mycov12(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./trained/model/myconv12/"

if builder == 'myconv01':
    model_name = "myconv01"
    model = build_myconv01(512, 512, 3)
    model_path = "./trained/model/myconv01/"

if builder == 'myconv02':
    model_name = builder
    model = build_myconv01(512, 512, 3, filter_rate=2)
    model_path = "./trained/model/myconv02/"

if builder == 'myconv03':
    model_name = builder
    model = build_myconv03(512, 512, 3, filter_rate=1)
    model_path = "./trained/model/myconv03/"

if builder == 'unet':
    model_name = 'unet'
    model_path = "./trained/model/unet/"
    model = U_Net(512, 512, 3)

if builder == 'unet++':
    model_name = 'unetPlus'
    model_path = "./trained/model/unetPlus/"
    model = Nest_Net(512, 512, 3)
    batch_size = 32

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(optimizer="Adam",
              loss=bce_dice_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef])

if not os.path.exists(os.path.join(log_path, model_name)):
    os.makedirs(os.path.join(log_path, model_name))
tbCallBack = TensorBoard(log_dir=os.path.join(log_path, model_name),
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                         )
tbCallBack.set_model(parallel_model)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=30,
                                               verbose=0,
                                               mode='min',
                                              )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, model_name+".{epoch:02d}-{val_loss:.2f}.hdf5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min',
                                             )
callbacks = [check_point, early_stopping, tbCallBack]

batch_size = 32

while batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        parallel_model.fit(x_train, y_train, batch_size=batch_size, epochs=100000, verbose=1, shuffle=True,
                           validation_data=(x_valid, y_valid), callbacks=callbacks)
        break
    except tf.errors.ResourceExhaustedError as e:
        batch_size = int(batch_size / 2.0)
        print("\n> Batch size = {}".format(batch_size))

parallel_model.load_weights(os.path.join(model_path, model_name+".h5"))
parallel_model.compile(optimizer="Adam",
              loss=dice_coef_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef])
# start to test
testGenerator = ImageDataProvider("../data/bowl/test/*.tif")
x_test, y_test = testGenerator(144)  # 288/2
print(">> Test  data: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))
print(">> Test  mask: {} | {} ~ {}\n".format(y_test.shape, np.min(y_test), np.max(y_test)))
p_test = parallel_model.predict(x_test, batch_size=batch_size, verbose=1)
eva = parallel_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
IoU = compute_iou(y_test, p_test)
print("\nSetup: {}".format(model_name))
print(">> Testing dataset mIoU  = {:.2f}%".format(np.mean(IoU)))
print(">> Testing dataset mDice = {:.2f}%".format(eva[3]*100.0))
