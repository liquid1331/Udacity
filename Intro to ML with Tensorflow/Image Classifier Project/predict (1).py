import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt
import json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
# print(class_names)
import argparse as arg

from PIL import Image

def cf(s):
    with open(s, 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image/255
    image = image.numpy()
    return image

def predict(image_path, model, top_k=5):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    after_process = process_image(test_img)
    final_img = np.expand_dims(after_process, axis=0)
    preds = model.predict(final_img)
    p1=np.sort(preds[0])[::-1]
    c1=np.argsort(preds[0])[::-1]
    return ["Probabilities",p1[:top_k],"\n" , "class",c1[:top_k]]

def main():
    parser=arg.ArgumentParser()
    parser.add_argument('name', action="store", type=str)
    parser.add_argument('file', action="store",type=str)
    parser.add_argument('--top_k',type = int, default=1)
    parser.add_argument('--category_names',type=str,default="11")
    a=parser.parse_args()
    fun(a)

def fun (c):
    model = tf.keras.models.load_model(c.file,custom_objects={'KerasLayer':hub.KerasLayer})
    image_path=c.name
    if c.category_names=="11":
#         print(class_names)
        classes = predict(image_path,model,c.top_k)
#         print(classes)
        for i,b in zip(classes[4],classes[1]):
            print(f"{class_names[str(i+1)]} {b*100: .2f}%")
    else:
        c=cf(c.category_names)
        print(c)
if __name__ == "__main__":
    main()