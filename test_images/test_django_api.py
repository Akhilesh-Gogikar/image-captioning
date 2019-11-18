
# import the necessary packages
import requests
import cv2
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_attention(image, result, attention_plot):

    img_list = []
    cap_dict = {}

    with open("iu_xray.tsv") as f:
        for line in f:
            (key, value) = line.split("\t")
            #print(key)
            #print(value)
            if key == image:
                cap_dict[key] = str(value).replace(".","\n")
                #print(key)
                #print(value)

    imag = cv2.imread(image)
    temp_image = np.array(imag)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Ground_Truth:\n"+cap_dict[image])
    img = ax.imshow(temp_image)

    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("output", img)
    img_list.append(Image.fromarray(img))
    cv2.waitKey(0)

    result = result.split("[")[1]
    result = result.split("]")[0]
    result = result.split(",")
    #print(result)


    len_result = len(result)
    #for i in range(len_result):
        #print(result[i])

    #print(len_result)
    for l in range(len_result):
        fig = plt.figure(figsize=(10, 10))
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("AI_finding"+str(l)+":\n"+result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", img)
        img_list.append(Image.fromarray(img))
        cv2.waitKey(0)
    img_list[0].save((str(image.split(".")[0])+'.gif'),
                   save_all=True, append_images=img_list[1:], optimize=False, duration=2000, loop=0)

    #plt.tight_layout()
    #plt.show()

# define the URL to our face detection API
url = "http://localhost:8000/xray_attention/predict/"

#from __future__ import print_function


path = '.'

image_list = []

files = os.listdir(path)
for name in files:
    if ".png" in name:
        image_list.append(name)

print(image_list)

#image_list = ["CXR296_IM-1354-1001.png", "CXR422_IM-2065-5001.png" , "CXR1825_IM-0535-1002.png"]

for image in image_list:

# load our image and now use the face detection API to find faces in
# images by uploading an image directly
    #img = cv2.imread(image)
    payload = {"image": open(image, "rb")}
    r = requests.post(url, files=payload).json()
    json_load = json.loads(r["attentions"])
    #print(json_load["attn"])
    res = r["caption"]
    #print(res)
    a_restored = np.asarray(json_load["attn"])
    plot_attention(image,res,a_restored)
#print
#"adrian.jpg: {}".format(r)

# show the output image
