import numpy as np
import gradio as gr
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import pandas as pd
import datetime


 # STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    return annotated_image

mp_face_mesh = mp.solutions.face_mesh
def face_mesh(rgb_image):
    global detector
    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
   
    global data 
    data = pd.DataFrame(detection_result.face_blendshapes[0])
    # print(data)
    

    return data

def save_csv_file():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"output_{formatted_time}.csv"
    data.to_csv(filename, index=False)
    return data
   
def plot_face_blendshapes_bar_graph(face_blendshapes_data):
    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_data["category_name"], face_blendshapes_data["score"])
    ax.set_xlabel('Score')
    ax.set_ylabel('Names')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()

    # Label each bar with values
    for i, score in enumerate(face_blendshapes_data["score"]):
        plt.text(score, i, f"{score:.4f}", va="top")

    return fig

def plot_face_blendshapes_bar_graph_interface(rgb_image):
    first_res = face_mesh(rgb_image)


    fig = plot_face_blendshapes_bar_graph(first_res)
    
    
    return fig


def clear_input_output():
    # Clear the input and output values
    im.clear()
    ou.clear()

def handle_clear_click():
    clear_input_output()
# with gr.Blocks() as demo:
#     with gr.Row():   
#         gr.Interface(
#             fn=plot_face_blendshapes_bar_graph_interface, 
#             inputs=gr.inputs.Image(shape=None),
#             outputs="plot")
#     btn = gr.Button("Download")
#     btn.click(plot_face_blendshapes_bar_graph_interface)

with gr.Blocks() as demo:
    with gr.Row():
        im= gr.inputs.Image(shape=None)
        ou=gr.Plot()
    with gr.Row():
        with gr.Row():
            btn = gr.Button(value="Submit")
            btn.click(plot_face_blendshapes_bar_graph_interface, 
            inputs=[im],
            outputs=[ou])
            clear_btn = gr.Button(value="Clear")
            clear_btn.click(lambda x: gr.update(value=None), [], [im])
        with gr.Row():
            btn = gr.Button(value="Download")
            btn.click(save_csv_file)
            clear_btn_out = gr.Button(value="Clear")
            clear_btn_out.click(lambda x: gr.update(value=None), [], [ou])


demo.launch()
# iface.launch()
