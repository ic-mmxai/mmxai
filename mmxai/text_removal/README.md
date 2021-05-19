# Inpainting module

## Library Requirements
1. numpy
2. OpenCV (cv2)
3. PIL
4. math
5. request

## Model requirement
For smart text removal, user need to manually download the pretrained EAST detection model
from this [dropbox link](https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1), decompress it and save it under the text_removal folder. The relative modal path
should be "text_removal/frozen_east_text_detection.pb".

## Example Usage
### white text removal
```python
from white_text_removal import removeText

img_path = "https://drivendata-public-assets.s3.amazonaws.com/memes-overview.png"
img = removeText(img_path, threshold=254)
img.show()
```

### smart text removal

Smart text removal is developed from [this example](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py)

```python
from smart_text_removal import SmartTextRemover

remover = SmartTextRemover("text_removal/frozen_east_text_detection.pb")
img = remover.inpaint(
    "https://www.iqmetrix.com/hubfs/Meme%2021.jpg")
img.show()
```