import streamlit as st
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo

# Fungsi untuk memuat model yang telah dilatih
@st.cache_resource
def load_trained_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "E:/Mask R-CNN BRIN/detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/model_final (1).pth"  # Path ke model yang telah dilatih
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Sesuaikan dengan jumlah kelas dalam dataset kamu
    
    # Register dataset dan metadata
    welding_metadata = {
        "thing_classes": ["Spatter", "slag inclusion", "spatter"]
    }
    MetadataCatalog.get("welding_defects").set(**welding_metadata)
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

predictor, cfg = load_trained_model()

def get_class_name(class_id):
    class_map = {0: "Spatter", 1: "slag inclusion", 2: "spatter"}
    return class_map.get(class_id, "Unknown")

def main():
    with st.expander("About the App"):
        st.markdown('<p style="font-size: 30px;"><strong>Welcome to my Instance Segmentation App!</strong></p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 20px; color: white;">Aplikasi ini dibuat menggunakan Streamlit, library Detectron2, dan OpenCV untuk melakukan <strong>Instance Segmentation</strong> pada video dan gambar.</p>', unsafe_allow_html=True)

    option = st.selectbox(
        'Apa tipe file yang ingin kamu deteksi?',
        ('Images', 'Videos')
    )

    if option == "Images":
        st.title('Instance Segmentation Welding Defect for Images With Mask R-CNN')
        st.subheader("""
        Ini akan memasukkan gambar dan mengeluarkan dengan memberikan garis besar objek cacat pada gambar.
        """)
        uploaded_file = st.file_uploader("Unggah gambar...", type="jpg")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            st.write(file_details)

            image = cv2.imdecode(file_bytes, 1)

            outputs = predictor(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("welding_defects"), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]

            st.image(image, caption='Gambar Asli', use_column_width=True)
            st.image(result_image, caption='Gambar Hasil Deteksi.', use_column_width=True)

            pred_classes = outputs["instances"].pred_classes
            pred_boxes = outputs["instances"].pred_boxes

            st.write("Deteksi objek:")
            for i, class_id in enumerate(pred_classes):
                st.write(f"Objek {i+1}: {get_class_name(class_id.item())}")
                st.write(f"Bounding Box: {pred_boxes[i].tensor.tolist()}")
    else:
        st.title('Instance Segmentation Welding Defect for Videos With Mask R-CNN')
        st.subheader("""
        Ini akan memasukkan Video dan mengeluarkan dengan memberikan garis besar objek cacat pada Video.
        """)
        # func_2('instanceSegmentation')

if __name__ == '__main__':
    main()
