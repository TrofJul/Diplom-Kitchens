import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from IPython.utils import io
import wget

import torch
from torchvision import transforms, models
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from collections import OrderedDict
from matplotlib import pyplot as plt

import requests

from colordetect import ColorDetect
import webcolors

import sys, os
sys.path.append('models/research')
sys.path.append('models/research/object_detection')

from time import process_time
import six.moves.urllib as urllib
from collections import Counter
import tarfile
import zipfile
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#Вспомогательные функции для скачивания файлов

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

download_file_from_google_drive('1omZW_rWygbLvvQ399PnmD-LdZHOxpSZ_', 'weights.pth')
                                 
#download_file_from_google_drive('1ipDvv0tPxu63GrmVM0yQ5E7ispL7HEi_', 'label_encoder.pkl')
#download_file_from_google_drive('1ipDvv0tPxu63GrmVM0yQ5E7ispL7HEi_', 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz')


#Детекция объектов

model_path = 'http://download.tensorflow.org/models/object_detection/'
model_name = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'

if not os.path.exists(model_name + '.tar.gz'):
  mod = wget.download(model_path + model_name + '.tar.gz')
tar = tarfile.open(model_name + '.tar.gz', 'r:gz')
model_file_name =  model_name + '/frozen_inference_graph.pb'
tar.extract(model_file_name)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_file_name, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = {
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    67: {'id': 67, 'name': 'dining table'},
    72: {'id': 72, 'name': 'tv'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'}
}

translation = {
    'chair' : 'стул',
    'couch':  'диван',
    'dining table': 'кухонный стол',
    'tv': 'телевизор',
    'microwave': 'микроволновая печь',
    'oven': 'духовка',
    'toaster': 'тостер',
    'sink': 'раковина',
    'refrigerator': 'холодильник'
}


def load_image(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Запуск поиска объектов
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # Преобразование выходных данных из массивов float32 в нужный формат
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

def filter_output(output_dict):
    my_categories = category_index.keys()
    for i in range(len(output_dict['detection_classes'])):
        if not (output_dict['detection_classes'][i] in my_categories):
            output_dict['detection_classes'][i] = 1
            output_dict['detection_scores'][i] = 0
            output_dict['detection_boxes'][i] = [0., 0., 0., 0.]

def count_objects(output_dict):
    init_dict = {62: 0, 63: 0, 67: 0, 72: 0, 78: 0, 79: 0, 81: 0, 82: 0}
    filtered_output = []

    for i in range(len(output_dict['detection_classes'])):
      if output_dict['detection_classes'][i] in category_index.keys() and output_dict['detection_scores'][i] > 0.5:
        filtered_output.append(output_dict['detection_classes'][i])
    
    counted_output = dict( list(init_dict.items()) + list(dict(Counter(filtered_output)).items()))
    result = counted_output.copy()
    for i in counted_output:
      result[category_index[i]['name']] = result.pop(i)
    return result

def classify(counted_objects):
    if counted_objects['oven'] > 0 or counted_objects['sink'] > 0 or counted_objects['refrigerator'] > 0:
      placeholder_text.subheader('На фотографии действительно изображена *кухня*')
      st.write('\n')
      isKitchen = True
    else:
      placeholder_text.subheader('Кажется на этой фотографии изображена *не кухня*, загрузите другое фото')
      st.write('\n')
      isKitchen = False
    return isKitchen

def detect(jpg):
    kitchen = load_image(jpg)

    start_time = process_time() 
    output_dict = run_inference_for_single_image(kitchen, detection_graph)

    filter_output(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
          kitchen,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=6,
          min_score_thresh=.6)
    plt.figure(figsize=(20, 12))
    plt.grid(False)
    placeholder_im.image(kitchen)
    placeholder_time.write('detection time: ' + str(round(process_time()  - start_time)) + ' seconds')

    counted_objects = count_objects(output_dict)
    counted_objects_tr = {}

    for obj in counted_objects.keys():
      counted_objects_tr[translation[obj]] = counted_objects[obj]

    isKitchen = classify(counted_objects)
    if isKitchen:
      with right_column:
        for obj in counted_objects_tr.items():
          st.write(f'**{obj[0]}**' + ': ' + str(obj[1]))
    return isKitchen
''''''
#Классификация кухни по конфигурации

# разные режимы датасета 
DATA_MODES = ['train', 'val', 'test']
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224

class KitchenDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = np.asarray(file)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image.load()
        return image
  
    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)

def predict_one_sample(model, inputs, device='cpu'):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs

label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

vgg16 = models.vgg16(pretrained=True)
n_classes = 3

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, n_classes)),
                                        ('output', nn.LogSoftmax(dim=1))]))

vgg16.classifier = classifier
path_vgg16 = 'weights.pth'
vgg16.load_state_dict(torch.load(path_vgg16, map_location='cpu'))

''''''
#Распознавание основных цветов на изображении

def colortable(colors, title):
   
    width = 212
    height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40
   
    names = list(colors)
   
    length_of_names = len(names)
    length_cols = 1
    length_rows = length_of_names
   
    width2 = width * 2 + 2 * margin
    height2 = height * length_rows + margin + topmargin
    dpi = 72
   
    figure, axes = plt.subplots(figsize =(width2 / dpi, height2 / dpi), dpi = dpi)
    figure.subplots_adjust(margin / width2, margin / height2,
                        (width2-margin)/width2, (height2-topmargin)/height2)
      
    axes.set_xlim(0, width * 2)
    axes.set_ylim(height * (length_rows-0.5), -height / 2.)
    axes.set_axis_off()
    axes.set_title(title, fontsize = 24, loc ="left", pad = 10)
   
    for i, name in enumerate(names):
        rows = i % length_rows
        cols = i // length_rows
        y = rows * height
   
        swatch_start_x = width * cols
        swatch_end_x = width * cols + swatch_width
        text_pos_x = width * cols + swatch_width + 7
   
        axes.text(text_pos_x, y, name, fontsize = 14,
                horizontalalignment ='left',
                verticalalignment ='center')
   
        axes.hlines(y, swatch_start_x, swatch_end_x,
                  color = colors[name], linewidth = 18)
   
    plt.show()
    left_column.pyplot(figure)

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name

def detect_colors(img):
    width, height = img.size

    left = width/6
    top = height/6
    right = width - width/6
    bottom = height
    img = img.crop((left, top, right, bottom))
    img.save('img.jpg')

    user_image = ColorDetect('img.jpg')
    color_count = user_image.get_color_count(color_format="rgb")

    colors = {}
    for col in color_count.keys():
        rgb_str = col.strip('[]').split(',')
        rgb = []
        for s in rgb_str:
            rgb.append(int(float(s.strip())))
        rgb = tuple(rgb)
        colors[get_colour_name(rgb)] = tuple(map(lambda x: x/255, rgb))

    colortable(colors, "Основные цвета")



''''''
#Основная программа
#формирование web-страницы

#st.beta_set_page_config(page_title='Kitchens.io', page_icon = ':smiley:', layout = 'centered')
st.title("Определение параметров кухни")

image = None

uploaded_file = st.file_uploader('Выберите или перетащите файл с изображением')
if uploaded_file is not None:
  image = Image.open(uploaded_file)

url = st.text_input('Или введите url-адрес изображения кухни')
if url is not '':
  jpg = wget.download(url)
  image = Image.open(jpg)


if image is not None:
  placeholder_im = st.empty()
  placeholder_time = st.empty()
  placeholder_text = st.empty()
  left_column, right_column = st.beta_columns(2)
  with st.spinner('Пожалуйста, подождите...'):
    isKitchen = detect(image.convert('RGB'))
  if isKitchen:
    kitchen = KitchenDataset([image], mode="test")
    prob_pred = predict_one_sample(vgg16, kitchen[0].unsqueeze(0))
    predicted_proba = np.max(prob_pred)*100
    y_pred = np.argmax(prob_pred)
    predicted_label = label_encoder.classes_[y_pred]
    label_translation = {
        'corner_kitchen': 'угловая кухня',
        'straight_kitchen': 'прямая кухня',
        'island_kitchen': 'островная кухня'
    }
    with right_column:
      st.write('**Конфигурация**: ', label_translation[predicted_label])
    detect_colors(image)