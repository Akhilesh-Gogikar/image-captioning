#from django.shortcuts import render

# Create your views here.

# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from img2vec_pytorch.img_to_vec import Img2Vec
import numpy as np
import urllib
import json
import cv2
import os

import pandas as pd

from PIL import Image

# define the path to the face detector
#FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    #base_path=os.path.abspath(os.path.dirname(__file__)))




@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    img2vec = Img2Vec(cuda=True)

    train_data = pd.read_csv('/home/akhilesh/bmc_api/xray_caption/train_images.tsv', sep="\t", header=None)
    train_data.columns = ["id", "caption"]
    train_images = dict(zip(train_data.id, train_data.caption))

    ids = [i+1 for i in range(len(train_data.id))]

    raw = np.load("/home/akhilesh/bmc_api/xray_caption/raw_embeddings.npy")

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        vec = img2vec.get_vec(image)

        # Compute cosine similarity with every train image
        vec = vec / np.sum(vec)
        # Clone to do efficient mat mul dot
        test_mat = np.array([vec] * raw.shape[0])
        sims = np.sum(test_mat * raw, 1)
        top1 = np.argmax(sims)
        # Assign the caption of the most similar train image

        print("top1: " + str(top1))

        caption = train_images[ids[top1]]

        #detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        #rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          #minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # construct a list of bounding boxes from the detection
        #rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        data.update({"caption": str(caption), "success": True})

    # return a JSON response
    return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        print(path)
        image = Image.open(path)
        image = image.convert('RGB')

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        pixels = np.asarray(bytearray(data), dtype="uint8")
        print(pixels.shape)

        image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(img, 'RGB')

    # return the image
    return image
