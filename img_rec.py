import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

#load ResNet50 model
model = resnet50.ResNet50()

#load image, resizing to 224*224px
img = image.load_img("painting.jpg", target_size(224, 224))

#convert image to array
x = image.img_to_array(img)

#add a fourth dimension, because keras expects a list of images
x = np.expand_dims(x, axis=0)

#scale the input img to the range
x = reset50.preprocess_input(x)

#run the image to make a prediction
predictions = model.predict(x)
predicted_classes = resnet50.decode_predictions(predictions, top=5)

print("this image is...")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("{}: {2f} likelihood".format(name, likelihood))
