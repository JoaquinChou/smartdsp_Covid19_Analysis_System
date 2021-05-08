import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
# import time
import torch.nn.functional  as F
import numpy as np
import torch.nn as nn
# import math
import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


# # def Res_grade():
# #     return ResNet(ResidualBlock)
#
#
# def evaluate(model, criterion):
#     model.eval()
#     corrects = eval_loss = 0
#     j = 0
#     with torch.no_grad():
#         for image, label in testLoader:
#             # print(image.cpu().numpy().shape)
#             image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
#             label = Variable(label.cuda())  # 同理
#             pred = model(image)
#             loss = criterion(pred, label)
#
#             eval_loss += loss.item()
#             # print(loss.item())
#             max_value, max_index = torch.max(pred, 1)
#             pred_label = max_index.cpu().numpy()
#             true_label = label.cpu().numpy()
#             b1 = math.floor(test_sum / batch_size)
#
#
#             # print("***************")
#             corrects += np.sum(pred_label == true_label)
#             # print((torch.max(pred, 1)[1].view(label.size()).data == label.data).sum())
#     return eval_loss / float(len(testLoader)), corrects, corrects / test_sum, len(testLoader)
#
#
# def train(model, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     train_corrects = 0
#     num = 0
#     for i, (image, label) in enumerate(trainLoader):
#         # for i in range(image.size(0)):
#         #     b = random.uniform(0, 1)
#         #     if (b > 0.6):
#         #         im5 = np.transpose(image[i], (1, 2, 0))
#         #         im6 = skimage.util.random_noise(im5, mode='gaussian', seed=None, clip=True,
#         #                                         var=random.uniform(0.01, 0.05))
#         #         image[i] = transforms.ToTensor()(im6)
#
#         image = Variable(image.cuda())  # 同理
#         label = Variable(label.cuda())  # 同理
#         optimizer.zero_grad()
#
#         target = model(image)
#         loss = criterion(target, label)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         max_value, max_index = torch.max(target, 1)
#         pred_label = max_index.cpu().numpy()
#         true_label = label.cpu().numpy()
#         train_corrects += np.sum(pred_label == true_label)
#
#
#
#         num += 1
#         # print(num)
#
#     return total_loss / float(len(trainLoader)), train_corrects / train_sum


# def main():
#     import matplotlib.pyplot as plt
#     # model = models.resnet101(pretrained=True) #pretrained表示是否加载已经与训练好的参数
#     # model.fc =torch.nn.Linear(2048, 2) #将最后的fc层的输出改为标签数量
#     model = models.resnet34(pretrained=True)
#     # model = resnet34()
#     model = model.cuda()  # 如果有GPU，而且确认使用则保留；如果没有GPU，请删除
#     criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
#     # criterion = FocalLoss(2)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 定义优化函数
#
#     train_loss = []
#     valid_loss = []
#     accuracy = []
#     f1_max = 0
#     acc_temp = 0
#
#     # model.load_state_dict(torch.load('AGE_1.pth'), strict=True)
#     try:
#         # print('-' * 90)
#         for epoch in range(1, epoches + 1):
#             epoch_start_time = time.time()
#             loss, train_acc = train(model, optimizer, criterion)
#             # train_loss.append(loss * 1000.)
#             train_loss.append(loss)
#
#             print('| start of epoch {:3d} | time: {:2.2f}s | train_loss {:5.6f}  | train_acc {}'.format(epoch,
#                                                                                                         time.time() - epoch_start_time,
#                                                                                                         loss,
#                                                                                                         train_acc))
#             with open('log1_copy.txt', 'a') as f:
#                 f.write('| start of epoch {:3d} | time: {:2.2f}s | train_loss {:5.6f}  | train_acc {}'.format(epoch,
#                                                                                                         time.time() - epoch_start_time,
#                                                                                                         loss,
#                                                                                                         train_acc))
#                 f.write('\n')
#
#
#             loss, corrects, acc, size = evaluate(model, criterion)
#
#
#             valid_loss.append(loss)
#             accuracy.append(acc)
#
#             if acc > acc_temp:
#                 print(str(epoch)+"epoch:  save resnet_v2.pth model")
#                 torch.save(model.state_dict(), save_model_path+'resnet_v2.pth')
#                 with open('log1.txt', 'a') as f:
#                     f.write(str(epoch)+'epoch: save resnet_v2.pth model')
#                 acc_temp = acc
#
#
#
#
#             print('-' * 10)
#             print('| end of epoch {:3d} | time: {:2.2f}s | test_loss {:.6f} | accuracy {}'.format(epoch,
#                                                                                                   time.time() - epoch_start_time,
#                                                                                                   loss,
#                                                                                                   acc
#                                                                                                   ))
#
#             print('-' * 10)
#
#
#     except KeyboardInterrupt:
#         print("-" * 90)
#     # torch.save(model_object, 'model.pkl')
#     print("**********ending*********")
#     plt.plot(train_loss)
#     plt.plot(valid_loss)
#     plt.title('loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     # plt.savefig("C:/Users/901/Desktop/models1/mymodel_1_grade_loss.jpg")
#     # plt.cla()
#     plt.show()
#     plt.plot(accuracy)
#     plt.title('acc')
#     plt.ylabel('acc')
#     plt.xlabel('epoch')
#     # plt.savefig("C:/Users/901/Desktop/models1/mymodel_1_grade_acc.jpg")
#     plt.show()


def pred():
    model = models.resnet34()
    # model = resnet34()
    # model = model.cuda()   ####################
    # model.load_state_dict(torch.load('./model/resnet_v2.pth'), strict=True)##########################
    model.load_state_dict(torch.load('resnet_v2.pth', map_location=lambda storage, loc: storage), strict=True)
    model.eval()
    i = 0
    corrects = 0
    num = 0
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=False)
    for i, (image, label) in enumerate(testLoader):
        # image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
        image = Variable(image)
        pred = model(image)
        max_value, max_index = torch.max(pred, 1)
        pred_label = max_index.cpu().numpy()
        probs = F.softmax(pred, dim=1)
        # print("Sample probabilities:\n", probs[:2].data.detach().cpu().numpy())
        a, b = np.unravel_index(probs[:2].data.detach().cpu().numpy().argmax(),
                                probs[:2].data.detach().cpu().numpy().shape)  # 索引最大值的位置   ###b就说预测的label
        print(testLoader.dataset.imgs[i][0])

        print('预测结果的概率:', round(probs[:2].data.detach().cpu().numpy()[0][b] * 100))
        print("label:  "+str(b))
    #############################################################已知结果时才能预测
        pred_label = max_index.cpu().numpy()
        # print(pred_label)
        true_label = label.cpu().numpy()
        # corrects += np.sum(pred_label == true_label)   ####################################################################################################
        corrects += np.sum(pred_label == 2)
        print(corrects)

        num += 1
        print(num)
        print('------------------------------------------')


def predict_demo(mode):
    batch_size = 1
    path_test = './temp/'
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),  # 将图片数据变为torchtensor格式
    ])  # 对数据进行normalize
    testData = dsets.ImageFolder(path_test, transform=transform_test)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
    model = models.resnet50()
    if mode == 1:
        model.load_state_dict(torch.load('resnet_cxr.pth', map_location=lambda storage, loc: storage), strict=True)
        labels = ['COVID-19', 'NORMAL', 'Viral Pneumonia']
    else:
        model.load_state_dict(torch.load('resnet_ct.pth', map_location=lambda storage, loc: storage), strict=True)
        labels = ['COVID', 'non-COVID']



    # model.load_state_dict(torch.load('resnet50_v2.pth', map_location=lambda storage, loc: storage), strict=True)
    model.eval()
    for i, (image, label) in enumerate(testLoader):
        # image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
        image = Variable(image)
        pred = model(image)
        max_value, max_index = torch.max(pred, 1)
        pred_label = max_index.cpu().numpy()
        probs = F.softmax(pred, dim=1)
        # print("Sample probabilities:\n", probs[:2].data.detach().cpu().numpy())
        a, b = np.unravel_index(probs[:2].data.detach().cpu().numpy().argmax(),
                                probs[:2].data.detach().cpu().numpy().shape)  # 索引最大值的位置   ###b就说预测的label
        print(testLoader.dataset.imgs[i][0])
        print('预测结果的概率:', round(probs[:2].data.detach().cpu().numpy()[0][b] * 100))
        print("label:  " + str(b))
        #############################################################已知结果时才能预测
        pred_label = max_index.cpu().numpy()
        # print(pred_label)
        # labels = ['COVID-19','NORMAL','Viral Pneumonia']
        print(labels[int(b)])
    return str(round(probs[:2].data.detach().cpu().numpy()[0][b] * 100)), labels[int(b)]










# if __name__ == "__main__":
    # batch_size = 8  # 8   test:1
    # learning_rate = 1e-4  # 4  # 1e-5
    # epoches = 100  # 500
    # # save_model_path = './model/'
    # #     if not os.path.exists(save_model_path):
    # #         os.makedirs(save_model_path)
    #
    #
    # path_train ="./train2/"
    # # path_test = './val2/'
    # # path_test = './test2/'
    # path_test = './temp/'
    # # path_test = './Refuge2Validation/'
    #
    # transform_train = transforms.Compose([
    #     # transforms.RandomRotation(20),
    #     # transforms.ColorJitter(brightness=0.1),
    #     # transforms.ColorJitter(contrast=0.1),
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),  # 将图片数据变为torchtensor格式
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #     #                      std=[0.229, 0.224, 0.225]),
    # ])  # 对数据进行normalize
    #
    # transform_test = transforms.Compose([
    #     # transforms.ColorJitter(brightness=0.1),
    #     # transforms.ColorJitter(contrast=0.1),
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),  # 将图片数据变为torchtensor格式
    # ])  # 对数据进行normalize
    #
    # # transform = transforms.Compose([transforms.ToTensor() ]) #对数据进行normalize
    #
    # trainData = dsets.ImageFolder(path_train, transform=transform_train)  # 读取训练集，标签就是train目录下的文件夹的名字，图像保存在格子标签下的文件夹里
    # testData = dsets.ImageFolder(path_test, transform=transform_test)
    #
    # trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    # testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
    # test_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(path_test))])
    # train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(path_train))])
    #
    # name = []
    # # main()
    #
    # batch_size = 1
    # pred()


