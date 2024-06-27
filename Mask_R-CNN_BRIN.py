import streamlit as st
import cv2
import numpy as np
import torch
import detectron2
import os
import sys
import subprocess
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="Welding Defect Prediction with Mask R-CNN",
    page_icon="👁‍🗨",
    layout="wide"
)

st.title("👁‍🗨 Welding Defect Prediction with Mask R-CNN")

# Tentukan path ke direktori detectron2 di proyek Anda
detectron2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'detectron2'))

# Tambahkan path ke sys.path
sys.path.insert(0, detectron2_path)

try:
    import detectron2
except ImportError:
    print("Detectron2 not found, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
    import detectron2

@st.cache_resource
def load_trained_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "E:/Mask_R-CNN/detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/model_final (1).pth"  # Path ke model yang telah dilatih
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

def process_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Video file could not be opened.")
        return None
    
    output_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get("welding_defects"), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_frame = out.get_image()[:, :, ::-1]
        output_frames.append(result_frame)
    
    cap.release()
    return output_frames

def main():
    with st.expander("About the App"):
        st.markdown('<p style="font-size: 30px;"><strong>Welcome to my Instance Segmentation App!</strong></p>', unsafe_allow_html=True)
        st.markdown('''<p style="font-size: 20px; color: white;">Aplikasi ini dibuat menggunakan Streamlit, library Detectron2, dan OpenCV untuk melakukan <strong>Instance Segmentation</strong> pada video dan gambar.
                    Aplikasi ini digunakan untuk mendeteksi cacat pada pengelasan dengan memproses gambar pengelasan dan menerapkan algoritma Mask R-CNN. Teknologi ini memungkinkan identifikasi dari berbagai jenis cacat seperti spatter dan juga slag inclusion, yang dapat diidentifikasi secara langsung pada gambar.
Aplikasi ini merupakan pengembangan dari teknologi computer vision yang memanfaatkan jaringan saraf konvolusional (CNN) untuk segmentasi instance, memungkinkan identifikasi dan anotasi detail cacat pada gambar pengelasan.</p>''', unsafe_allow_html=True)

    option = st.selectbox(
        'Apa tipe file yang ingin kamu deteksi?',
        ('🔍 Overview', '🖼️ Images', '📽️ Videos')
    )

    if option == "🔍 Overview":
        st.title('Rangkuman')
        st.subheader("Mask R-CNN")
        st.write('Mask R-CNN (Mask Region-based Convolutional Neural Network) adalah algoritma deep learning yang digunakan untuk melakukan instance segmentation pada gambar. Instance segmentation adalah teknik yang tidak hanya mendeteksi objek dalam gambar, tetapi juga menentukan bentuk spesifik (mask) dari setiap objek terdeteksi. Ini merupakan pengembangan lebih lanjut dari algoritma Faster R-CNN, yang hanya melakukan object detection dan menghasilkan bounding box tanpa menentukan mask objek.')
        st.image('Mask R-CNN.png', caption='Spatter', width=500)
        st.subheader('Spatter')
        st.write('Spatter adalah salah satu jenis cacat pengelasan yang terjadi ketika terdapat percikan logam cair yang mengeras dan menempel di sekitar sambungan pengelasan. Ini dapat disebabkan oleh beberapa faktor, termasuk pengaturan pengelasan yang tidak tepat, seperti arus yang terlalu tinggi atau kecepatan pengelasan yang terlalu rendah, serta penggunaan parameter pengelasan yang tidak sesuai dengan jenis logam atau ketebalan material yang dielas. Spatter dapat mengurangi estetika dari sambungan pengelasan dan dapat menyebabkan masalah dalam proses penggabungan atau perakitan, karena percikan logam yang menempel dapat mengganggu penggabungan komponen atau mempengaruhi operasi mesin. Selain itu, spatter juga dapat menambah biaya produksi karena memerlukan pembersihan tambahan dan dapat mengurangi efisiensi penggunaan bahan baku')
        st.image('Spatter.png', caption='Spatter', width=500)
        st.subheader('Slag Inclusion')
        st.write('Slag inclusion merupakan kondisi di mana material slag tertinggal dalam sambungan pengelasan. Hal ini bisa terjadi karena kurangnya pembersihan yang sempurna atau kurangnya perlindungan gas yang cukup. Keberadaan slag dapat mengurangi kekuatan sambungan dan berpotensi menciptakan titik lemah dalam struktur. Secara umum, slag akan muncul di permukaan logam las karena perbedaan tegangan permukaan dan densitas yang lebih rendah, namun bisa juga terperangkap di bawah permukaan atau terjebak di dalam celah sambungan. Ketidakbersihan permukaan logam sebelum pengelasan, adanya partikel slag yang belum terhapus sepenuhnya, serta pemilihan elektroda atau bahan pengisi yang tidak cocok dengan logam dasar dapat menjadi penyebab terjadinya slag inclusion')
        st.image('Slag Inclusion.png', caption='Slag Inclusion', width=500)
    
    elif option == "🖼️ Images":
        st.title('🖼️ Instance Segmentation Welding Defect for Images With Mask R-CNN')
        st.subheader("""
        Ini akan memasukkan gambar dan mengeluarkan dengan memberikan garis besar objek cacat pada gambar.
        """)
        uploaded_files = st.file_uploader("Unggah gambar...", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
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

                # Convert the result image to PIL format
                result_image_pil = Image.fromarray(result_image)
                buf = BytesIO()
                result_image_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                # Add a download button
                st.download_button(
                    label="Unduh gambar hasil deteksi",
                    data=byte_im,
                    file_name=f"{uploaded_file.name.split('.')[0]}_detected.png",
                    mime="image/png"
                )
    elif option == "📽️ Videos":
        st.title('📽️ Instance Segmentation Welding Defect for Videos With Mask R-CNN')
        st.subheader("""
        Ini akan memasukkan Video dan mengeluarkan dengan memberikan garis besar objek cacat pada Video.
        """)
        uploaded_video = st.file_uploader("Unggah video...", type=['mp4', 'mov', 'avi', 'mkv'])

        if uploaded_video is not None:
            file_bytes = np.asarray(bytearray(uploaded_video.read()), dtype=np.uint8)
            video_path = 'temp_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(file_bytes)

            st.video(video_path)

            output_frames = process_video(video_path, predictor)

            if output_frames:
                out_video_path = 'output_video.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width, layers = output_frames[0].shape
                video = cv2.VideoWriter(out_video_path, fourcc, 20, (width, height))

                for frame in output_frames:
                    video.write(frame)

                video.release()

                with open(out_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                st.download_button(
                    label="Unduh video hasil deteksi",
                    data=video_bytes,
                    file_name="output_video.mp4",
                    mime="video/mp4"
                )

if __name__ == '__main__':
    main()
