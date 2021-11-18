import matplotlib.pyplot as plt
import cv2
import numpy as np


def imshow(image, grayscale=False):
    try:
        plt.imshow(image, cmap='gray' if grayscale else None)
    except:
        plt.imshow(image[0], cmap='gray' if grayscale else None)
    plt.axis('off')
    plt.show()


def cv2_imshow(image, grayscale=False):
    if not grayscale:
        cvtedimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        cvtedimage = image
    imshow(cvtedimage, grayscale)


def subplot_imshow(images, rows, columns, title='', grayscale=False):
    fig, ax = plt.subplots(rows, columns)

    fig.suptitle(title)

    i = 0
    try:
        for row in ax:
            try:
                for col in row:
                    try:
                        col.imshow(
                            images[i], cmap='gray' if grayscale else None)
                    except:
                        col.imshow(
                            images[i][0], cmap='gray' if grayscale else None)
                    col.axis('off')
                    i += 1
            except TypeError:
                try:
                    row.imshow(
                        images[i], cmap='gray' if grayscale else None)
                except:
                    row.imshow(
                        images[i][0], cmap='gray' if grayscale else None)
                row.axis('off')
                i += 1
            except IndexError:
                break
    except TypeError:
        imshow(images[i], grayscale)


def subplot_cv2_imshow(images, rows, columns, title='', grayscale=False):
    cvtedImg = []
    for image in images:
        if grayscale:
            cvtedImg.append(image)
        else:
            cvtedImg.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    subplot_imshow(cvtedImg, rows, columns, title, grayscale)


def pytorch_imshow(imagetensor, grayscale=False):
    while (imagetensor.dim() > 3):
        imagetensor = imagetensor.squeeze(0)
    imshow(imagetensor.numpy().transpose(1, 2, 0), grayscale)


def tf_imshow(imagetensor, denormalize=False):
    if denormalize:
        image = ((imagetensor.numpy()+1.0)*127.5).astype(np.uint8)
        while (image.ndim > 3):
            image = image[0]

    else:
        image = imagetensor.numpy().astype(np.uint8)
        while (image.ndim > 3):
            image = image[0]

    imshow(image)


def subplot_tf_imshow(imagetensors, rows, columns, title='', denormalize=False):
    numpyImg = []
    for image in imagetensors:
        if denormalize:
            image_ = ((image.numpy()+1.0)*127.5).astype(np.uint8)
            while (image_.ndim > 3):
                image_ = image_[0]

        else:
            image_ = image.numpy().astype(np.uint8)
            while (image_.ndim > 3):
                image_ = image_[0]

        numpyImg.append(image_)
    subplot_imshow(numpyImg, rows, columns, title)
