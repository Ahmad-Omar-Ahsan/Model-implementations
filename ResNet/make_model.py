from model import ResNet

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet([3, 8, 36, 3], img_channel, num_classes)