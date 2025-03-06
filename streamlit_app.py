import os
import time
import psutil
import base64
import subprocess
import numpy as np
from PIL import Image
import streamlit as st
from g4f.client import Client
from image_processor import ImageProcessor

image_processor = None
client = Client()
MODEL = 'gpt-4o-mini'

models_path = 'models' # folder with models
data_path = 'data' # folder with csv data
defoult_image_path = os.path.join(data_path, 'test_image.jpg') # defolt image to use


@st.cache_resource()
def initialize_image_processor():
    """
    Initialize class for image procesing<br>
    """
    print('initialize image_processor')
    return ImageProcessor()

@st.cache_resource()
def call_chatGPT(base64_image):
    response = client.chat.completions.create(model = MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful  assistant that responds in Markdown."},
                    {"role": "user", "content": [
                    {"type": "text", "text": "Could you tell what food are on the image and their calories? Show the alarm if food is unhealsy or includes alcohol"},
                        {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]}
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
    return response.choices[0].message.content

 
def get_gpu_memory():
    """
    Get info about GPU usage (not usuful for this app because models run on onnx)<br>
    """
    result = subprocess.check_output(
                                    [
                                        'nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,nounits,noheader'
                                    ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]


def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

def main():
    global image_processor

    st.title("Food Analyzer")
            
    st.markdown('''#### Load image''')
    image_file = st.file_uploader('Upload an image file',type = ['png', 'jpg', 'jpeg']) 

    if image_file: 
        base64_image = encode_image(image_file)
        test_image = Image.open(image_file)
        test_image = np.array(test_image)  # Convert to NumPy
        image_shape = test_image.shape
    
        if image_shape[0] > image_shape[1]:
            st.image(test_image, width=int(image_shape[1] * (300/image_shape[0])))
        else:
            st.image(test_image, width=300)
            
        if st.button("Tell the total calories using Chat GPT"):
            result = call_chatGPT(base64_image)
            
            st.markdown(result)
            
            
        if st.button("Tell the total calories using CLIP and SegmentClip"):
        
            try:
                start_init_time = time.time()
                image_processor = initialize_image_processor()
                st.markdown(f"**Model init in {time.time() - start_init_time} seconds**") 
            except Exception as e:
                st.text(e)
            
            if image_processor and image_processor.clip is not None and \
                                   image_processor.segclip is not None:
                st.markdown('**Models was loaded**')
            else:
                st.markdown('**Models was not loaded**')   
                initialize_image_processor.clear()
            
            start_model_time = time.time()
            results = image_processor.apply_models(test_image)
            st.markdown(f"**Model run in {time.time() - start_model_time} seconds**")
            
            st.markdown("**Image contains:**")
            alarms = []
            for res in results:
                st.markdown(f"**{res}**: {(results[res][0] * results[res][1])//1000} calories")
                if results[res][-1] in ['junk food', 'alcohol', 'dessert'] and \
                      results[res][-1] not in alarms:
                    alarms.append(results[res][-1])
            if alarms:
                st.markdown(f"**There are unhealthy food: {str(alarms)}**")
    
    
    st.markdown('''---''')
    st.subheader("System Stats")
    st1, st2, st3 = st.columns(3)

    with st1:
        st.markdown(f"**RAM Memory usage: {psutil.virtual_memory()[2]} %**")

    with st2:
        st.markdown(f"**CPU Usage: {psutil.cpu_percent()} %**")

    with st3:
        try:
            st.markdown(f"**GPU Memory Usage: {get_gpu_memory()} MB**")
        except:
            st.markdown("**GPU Memory Usage: NA**")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("System exited")
