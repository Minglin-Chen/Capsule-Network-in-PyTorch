import torchvision
import torchvision.transforms as T

import numpy as np
from cv2 import cv2

if __name__=='__main__':

    dataset_rotate = torchvision.datasets.MNIST(
            root='../data/',
            train=True,
            transform=T.Compose([
                T.RandomAffine(degrees=180., translate=(2.0/28.0, 2.0/28.0)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)) ]), 
            download=True)

    dataset_no_rotate = torchvision.datasets.MNIST(
            root='../data/',
            train=True,
            transform=T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]), 
            download=True)

    for i, (rotate_item, no_rotate_item) in enumerate(zip(dataset_rotate, dataset_no_rotate)):

        images_rotate, labels_rotate = rotate_item
        images_no_rotate, labels_no_rotate = no_rotate_item

        print('images rotate {} {}~{}'.format(images_rotate.shape, images_rotate.min(), images_rotate.max()))
        print('labels rotate {}'.format(labels_rotate))
        print('images no rotate {} {}~{}'.format(images_no_rotate.shape, images_no_rotate.min(), images_no_rotate.max()))
        print('labels no rotate {}'.format(labels_no_rotate))

        images_render = torchvision.utils.make_grid(images_rotate, nrow=1, normalize=True)
        images_render = images_render.permute(1,2,0).numpy()
        images_render = (images_render - images_render.min()) / (images_render.max() - images_render.min())
        images_render = images_render * 255
        images_render = images_render.astype(np.uint8)
        images_render = cv2.resize(images_render, dsize=None, fx=8.0, fy=8.0)

        cv2.imshow('images rotate', images_render)

        images_render = torchvision.utils.make_grid(images_no_rotate, nrow=1, normalize=True)
        images_render = images_render.permute(1,2,0).numpy()
        images_render = (images_render - images_render.min()) / (images_render.max() - images_render.min())
        images_render = images_render * 255
        images_render = images_render.astype(np.uint8)
        images_render = cv2.resize(images_render, dsize=None, fx=8.0, fy=8.0)

        cv2.imshow('images no rotate', images_render)

        cv2.waitKey()

