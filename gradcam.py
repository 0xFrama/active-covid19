import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[::-1, :, :]
    for i in range(3):
        preprocessed_img[i, :, :] = preprocessed_img[i, :, :] - means[i]
        preprocessed_img[i, :, :] = preprocessed_img[i, :, :] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (1, 0, 2)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def calculate_pr_rc(img, mask):
    # select corresponding groundtruth
    groundtruth = np.load('./Paziente 1_convex_202003061327250047ABD_00561.npy')
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    red_image = heatmap.copy() # Make a copy
    red_image[:,:,0] = 0
    red_image[:,:,1] = 0
    heatmap = red_image
    heatmap = np.float32(heatmap) / 255
    black_img = np.zeros((224,224,3), dtype='uint8')
    cam = heatmap + np.float32(img)
    black_cam = heatmap + np.float32(black_img)
    cam = cam / np.max(cam)
    black_cam = black_cam / np.max(black_cam)
    cam = cv2.resize(black_cam, (1055,672))
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))

    # Load the aerial image and convert to HSV colourspace
    image = cv2.imread("cam.jpg", cv2.IMREAD_UNCHANGED)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "red"
    lower_red = np.array([0,50,20])
    upper_red = np.array([5,255,255])
    lower_red_2 = np.array([175,50,20])
    upper_red_2 = np.array([180,255,255])

    # Mask image to only select browns
    mask_1 = cv2.inRange(hsv, lower_red, upper_red)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = mask_1 | mask_2

    # Change image to white where we found red
    image[mask > 0] = (255, 255, 255)
    image = image / 255
'''
    intersection = np.logical_and(groundtruth, image)
    union = np.logical_or(groundtruth, image)
    iou_score = np.sum(intersection) / np.sum(union)
    print('IoU is %s' % iou_score)
    image = 255 * image
    cv2.imwrite("black.jpg", image)
'''


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = load_resnet18() # models.resnet18(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["1"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    #img = np.float32(img) / 255
    img = np.float32(cv2.resize(img, (224, 224))) / 255 # from [0,255] to [0,1]
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)


