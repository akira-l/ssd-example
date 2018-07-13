from mxnet.gluon import nn
from mxnet import nd
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

cls_pred = class_predictor(5, 10)
cls_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = cls_pred(x)
print(y.shape)
#(2, 55, 20, 20)
#there are 5 anchor, 10 classes 55 = 5*(10+1) 1 is for background


def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

box_pred = box_predictor(10)
box_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = box_pred(x)
print(y.shape)

#(2, 40, 20, 20)
#10 anchors 4 position 40 = 4*10



