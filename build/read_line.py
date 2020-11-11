import os
import random

xmlfilepath = "./dataset/build/test/images/"
saveBasePath = "./dataset/build/test/"

# trainval_percent = 0  # 没有测试集
# train_percent = 0  # 也就是只有0.9用于训练，0.1用于验证
temp_png = os.listdir(xmlfilepath) # 返回一个列表吗，是的
print(temp_png)
ftrain = open(os.path.join(saveBasePath, 'test.txt'), 'w') # 1.先打开文件，没有文件新建一个文件 2.然后在写入 3. 最后在关闭文件
total = []
for png in temp_png:
    if png.endswith(".png"):
        png = png +'\n'
        ftrain.write(png)
ftrain.close()



# num = len(total_xml)
# list = range(num)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# print(trainval)
# train = random.sample(trainval, tr)
#
# print("train and val size", tv)
# print("traub suze", tr)
# ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
# ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
# ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
# fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
#
# for i in list:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftrain.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftest.write(name)
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
