import streamlit as st
import requests
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageColor
import numpy as np
import cv2
from matplotlib import colors
from collections import namedtuple

# Définition de la structure de données Label
Label = namedtuple( 'Label' , [
    'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# Liste des labels
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

# Création du dictionnaire id_category
id_category = { label[4] : label.category for label in labels }

# Votre fonction pour générer une image à partir du masque
def generate_img_from_mask(mask, colors_palette=['b','g','r','c','m','y','k','w']):
    img_seg = np.zeros((mask.shape[0],mask.shape[1],3), dtype='float')
    for cat in id_category.keys():
        img_seg[:,:,0] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[0]
        img_seg[:,:,1] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[1]
        img_seg[:,:,2] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[2]
    return Image.fromarray((img_seg * 255).astype(np.uint8))

# Votre fonction pour afficher le résultat du masque
def affichage_result_mask(img_path, mask, output_size=(2048, 1024)):
    # Convertir le masque en numpy array si ce n'est pas déjà le cas
    mask_array = np.array(mask)
    # Redimensionner le masque pour correspondre aux dimensions de l'image originale
    mask_resized = cv2.resize(mask_array, output_size, interpolation=cv2.INTER_NEAREST)
    # Générer l'image à partir du masque redimensionné
    mask_t = generate_img_from_mask(mask_resized)
    return mask_t

def recoloriser_masque(mask, category2labels, colors_palette=['b','g','r','c','m','y','k','w']):
    # Créer une image vide avec 3 canaux pour la couleur
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for category, labels in category2labels.items():
        cat_index = list(category2labels).index(category)  # Obtenir l'index de la catégorie
        color = colors.to_rgb(colors_palette[cat_index % len(colors_palette)])  # Utiliser la palette de couleurs
        for label in labels:
            # Appliquer la couleur aux pixels correspondants dans le masque
            color_mask[mask == label.id] = [int(c*255) for c in color]  # Convertir la couleur en valeurs RGB

    return color_mask

# Fonction pour créer la légende des couleurs
colors_to_names = {
    (0, 0, 255): 'bleu',
    (0, 128, 0): 'vert',
    (255, 0, 0): 'rouge',
    (0, 255, 255): 'cyan',
    (255, 0, 255): 'magenta',
    (255, 255, 0): 'jaune',
    (0, 0, 0): 'noir',
    (255, 255, 255): 'blanc'
}

# Fonction modifiée pour créer la légende des couleurs avec les noms
def create_color_legend(category2labels, colors_palette):
    legend = {}
    for i, (category, labels) in enumerate(category2labels.items()):
        # Obtenir le code RGB à partir de la palette de couleurs
        rgb_color = ImageColor.getcolor(colors_palette[i % len(colors_palette)], "RGB")
        # Obtenir le nom de la couleur à partir du dictionnaire
        color_name = colors_to_names.get(rgb_color, 'Inconnu')
        legend[category] = color_name
    return legend

# Affichage de la légende des couleurs
def display_color_legend(legend):
    st.write("Légende des couleurs:")
    for category, color in legend.items():
        # Utiliser un DataFrame pour un affichage sous forme de tableau
        st.write(f"{category} : ", color)

st.title('Application de Segmentation d\'Image')
mask_folder = "image/Masks"
image_folder = "image/Original"
image_files = os.listdir(image_folder)
selected_image = st.selectbox('Sélectionnez une image:', image_files)

if st.button('Prédire'):
    with open(os.path.join(image_folder, selected_image), 'rb') as f:
        img_bytes = f.read()

    response = requests.post('https://segmodelp8-d4de570078ae.herokuapp.com/predict/', files={'file': img_bytes})
    mask = response.json()['mask']

    # Convertir le masque en image PIL et l'afficher
    mask_image = affichage_result_mask(os.path.join(image_folder, selected_image), mask)
    st.image(mask_image, caption='Masque Prédit')

    legend = create_color_legend(category2labels, colors_palette = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
)
    display_color_legend(legend)

    # Ajout pour afficher l'image originale
    original_image = Image.open(os.path.join(image_folder, selected_image))
    st.image(original_image, caption='Image Originale')

    # Ajout pour recoloriser et afficher le masque associé
    mask_associated_file = selected_image.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    mask_associated = np.array(Image.open(os.path.join(mask_folder, mask_associated_file)))

    # Recoloriser le masque associé
    color_mask_associated = recoloriser_masque(mask_associated, category2labels, colors_palette=['b','g','r','c','m','y','k','w'])

    # Convertir le masque recolorisé en image PIL et l'afficher
    color_mask_associated_image = Image.fromarray(color_mask_associated)
    st.image(color_mask_associated_image, caption='Masque Associé Recolorisé')


if __name__ == "__main__":
  port = int(os.environ.get("PORT"))
  st.server.start_server(port=port)
