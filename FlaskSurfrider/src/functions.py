import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import json
import cv2
from src.utils import ops as utils_ops
from src.utils import label_map_util
from src.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def init_graph():
    PATH_TO_FROZEN_GRAPH = 'models/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


# fonction qui permet de récupérer le résultat de la prédiction sous forme de DataFrame
# result_dict : dictionnaire des resultats renvoyé par la fonction run_inference_for_single_image
# image_size : taille de l'image à prédire
# iou : score à partir duquel on va afficher les bounding_boxes
def get_result_in_df(result_dict, image_size, iou):
    result_df = pd.DataFrame({"detection_score" : result_dict["detection_scores"],"detection_classe" : result_dict['detection_classes']})
    detection_boxes_df = pd.DataFrame(result_dict['detection_boxes'], columns=['ymin', 'xmin', 'ymax', 'xmax'])
    result_df = pd.concat([result_df, detection_boxes_df], axis=1)
    result_df['detection_classe'] = result_df['detection_classe'].map({1 : "bottles", 2 : "fragments", 3 : "others"})
    result_df["ymin"] = result_df["ymin"] * image_size[0]
    result_df["xmin"] = result_df["xmin"] * image_size[1]
    result_df["ymax"] = result_df["ymax"] * image_size[0]
    result_df["xmax"] = result_df["xmax"] * image_size[1]
    result_df["height"] = result_df["ymax"] -  result_df["ymin"]
    result_df["width"] = result_df["xmax"] - result_df["xmin"]
    result_df["left"] = result_df["ymin"]
    result_df["top"] = result_df["xmin"]
    result_df = result_df[result_df["detection_score"] > iou]
    return result_df

# fonction qui permet de récupérer le résultat de la prédiction sous forme de dictionnaire
# result_df : DataFrame des resultats renvoyé par la fonction get_result_in_df
# image_size : taille de l'image à prédire
# image_path : le chemin où se trouve l'image
def get_result_in_dict(result_df, image_size, image_path):
    image_name = image_path.split("/")[-1]
    result_dict = {}
    result_dict["asset"] = {'format': image_name.split(".")[-1],
                            'id': image_name.split(".")[0],
                            'name': image_name,
                            'path': image_path,
                            'size': {'height': image_size[0], 'width': image_size[1]},
                            'state': 2,
                            'type': 1}
    bounding_boxes = []
    for index, row in result_df.iterrows():
        bounding_boxes.append({'boundingBox': {'height': row['height'],
                                                 'left': row['left'],
                                                 'top': row['top'],
                                                 'width': row['width']},
                                 'id': image_name.split(".")[0] + row['detection_classe'] + str(index),
                                 'points': [],
                                 'tags': [row['detection_classe']],
                                 'type': 'RECTANGLE',
                                 'score': row['detection_score']})
    result_dict["regions"] = bounding_boxes
    result_dict['version'] = '2.1.0'
    return result_dict

# fonction qui permet de sauvegarder le résultat de la prédiction en json
# result_dict : dictionnaire des resultats renvoyé par la fonction get_result_in_dict
# save_folder : le chemin du répertoire où enregistrer le fichier json
def save_result_in_json(result_dict, save_folder):
    if not os.path.exists(os.path.abspath(save_folder)):
        os.makedirs(os.path.abspath(save_folder))
    with open(os.path.abspath(save_folder) + "/result_" + result_dict["asset"]["id"]+".json", 'w') as f:
        json.dump(result_dict, f)

def display_pred(image_path, detection_graph):
    PATH_TO_LABELS = 'src/label.pbtxt'
    SAVE_FOLDER = 'static/json/'
    IOU = 0.5
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    image = Image.open("static/img/"+image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=IOU,
        line_thickness=int((image_np_expanded.shape[1] + image_np_expanded.shape[2])/400))
    result_df = get_result_in_df(output_dict, image_np.shape, IOU)
    result_dict = get_result_in_dict(result_df, image_np.shape, image_path)
    save_result_in_json(result_dict, SAVE_FOLDER)
    return image_np