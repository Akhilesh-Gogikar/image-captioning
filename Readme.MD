This project deploys bio-medical image captioning using 1-Nearest Neighbour approach.

The image embeddings are extracted using img2vec.

The embeddings are then compared with the train-set and nearest caption is retrieved.

commands to run on local file:

curl -X POST -F image=@CXR2886_IM-1288-0001-0001.png 'http://localhost:8000/xray_caption/detect/' ; echo ""

commands to run on image url:

curl -X POST 'http://localhost:8000/xray_caption/detect/' -d 'url=https://img.medscapestatic.com/pi/features/slideshow-slide/chest-radiographs/fig1.jpg' ; echo ""
