import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
# Importing our ML model using pickle
model= pickle.load(open('img_model.p','rb'))

st.title('Image Classifier using Machine Learning')
st.text('Upload the Image')

uploaded_file= st.file_uploader('Choose an Image...',type='jpg')

if uploaded_file is not None:
    img= Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')
    
    if st.button('PREDICT'):
        CATAGORIES=['Cars', 'Bikes', 'Bicycle']
        st.write('Result...')
        flat_data=[]
        img= np.array(img)
        img_resized= resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data= np.array(flat_data)
        y_out= model.predict(flat_data)
        y_out= CATAGORIES[y_out[0]]
        st.write(f'PREDICTED OUTPUT: {y_out}')
        
