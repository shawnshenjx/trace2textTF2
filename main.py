from getData import getDataTrain1, getDataTrain2
from train import train, decode
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

size_list = [4000,8000,16000,32000,64000,128000]
scale_list = [0, 2, 4]
max_length = 500
label_pad = 25
input_dim = 2
num_classes = 28
learning_rate = 0.001
data_split = None
batch_size = 64
EPOCHS  = 15
load_model = False
SAVE_PATH = 'res/'
monitor = 'val_loss'


def main_fake(input1, label1, input2, label2, size):

    scale = ''
    train(max_length, input_dim, num_classes, learning_rate,input1, label1,
              input2, label2, batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model)


def main_real(size, scale):
    input1, label1 = getDataTrain2(max_length, data_split)
    input2, label2 = input1, label1

    train(max_length, input_dim, num_classes, learning_rate,input1, label1,
              input2, label2, batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model)




if __name__ == "__main__":
    input1, label1 = getDataTrain1(max_length, label_pad, data_split)
    input2, label2 = input1, label1
    main_fake(input1, label1, input2, label2,256000)