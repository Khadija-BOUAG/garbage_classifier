
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import torch
from torchvision import transforms
import os
from PIL import Image
import pickle


app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
model = pickle.load(open('model.pkl', 'rb'))

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

def predict_image(img, model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)#to_device(img.unsqueeze(0), device)
    #xb = img[0]
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    nb = preds[0].item()
    result = classes[nb]
    return result


# In[ ]:


@app.route('/')
def home() :
    return render_template('try2.html')

UPLOAD_FOLDER = 'upload'

@app.route('/', methods = ['GET','POST'])
def upload_predict():
    if request.method =='POST':
        image_file = request.files['image']
        if image_file :
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            image = Image.open(image_location) #Path('UPLOAD_FOLDER/' + image_file.filename))
            example_image = transformations(image)

            pred = predict_image(example_image, model)
            return render_template('try2.html', prediction=pred) # , image_loc = image_file.filename)
    # return render_template('try2.html', prediction=None, image_loc = None)

if __name__ == '__main__':
    app.run()

