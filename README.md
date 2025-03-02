# **AI-Powered Vehicle Distance Estimation & Object Detection Using YOLOv5 and Deep Learning**  

## **Project Overview**  
This project presents an advanced **AI-driven vehicle distance estimation system** integrated with **real-time object detection**. Leveraging the **YOLOv5 model**, deep learning techniques, and computer vision methodologies, it accurately detects vehicles in traffic and estimates their distances. The system is trained using the **KITTI and FLIR datasets**, making it robust for diverse environments.  

This end-to-end solution includes **dataset preprocessing, model training, inference, visualization, and results analysis**, ensuring a seamless workflow from data acquisition to practical deployment. The integration of **automated CSV generation, dataset formatting for YOLO, and custom training scripts** optimizes performance for real-world applications such as **autonomous vehicles, intelligent traffic monitoring, and road safety analysis**.  

## **Tech Stack & Tools Used**  
- **Programming Languages**: Python  
- **Deep Learning Frameworks**: PyTorch, TensorFlow  
- **Computer Vision**: OpenCV  
- **Object Detection**: YOLOv5  
- **Datasets**: KITTI, FLIR  
- **Model Training & Fine-Tuning**: Transfer Learning, Custom YOLOv5 Training  
- **Data Processing**: NumPy, Pandas  
- **Visualization & Results**: Matplotlib, OpenCV  

---

## **Implementation & Methodology**  

### **1. Data Acquisition & Preprocessing**  
- **Cloned & organized the KITTI and FLIR datasets** to train the distance estimation and object detection models.  
- Converted raw dataset annotations into **YOLOv5 format** to ensure compatibility.  
- Implemented **custom scripts** to **automatically download, preprocess, and structure** the dataset.  
- Generated a structured **CSV file** containing vehicle detection and distance labels.  

### **2. Model Training & Optimization**  
- Fine-tuned **YOLOv5 on the FLIR dataset** using a custom dataset YAML configuration.  
- Trained a **distance estimation model** using a pre-trained deep learning model with optimized hyperparameters.  
- Leveraged **transfer learning techniques** to improve accuracy and reduce training time.  
- Employed **augmentation strategies** to enhance the model’s robustness to various lighting and weather conditions.  

### **3. Inference & Prediction**  
- Developed a **real-time object detection pipeline** capable of detecting vehicles in video streams and estimating distances.  
- Applied **bounding box regression and confidence scoring** to enhance prediction accuracy.  
- Enabled **batch processing** of video frames for efficient real-time analysis.  

### **4. Result Visualization & Performance Analysis**  
- Integrated **OpenCV & Matplotlib** to visualize detected objects and estimated distances.  
- Automated **video output generation** for reviewing model performance.  
- Implemented a **coordinate mapping system** to export detected objects' positions into structured CSV files.  

---

## **Innovations & Unique Contributions**  
✅ **Enhanced YOLOv5 Fine-Tuning**: Unlike conventional object detection projects, this system **specifically fine-tunes YOLOv5 on FLIR thermal data**, improving detection in **low-light and foggy conditions**.  
✅ **Custom Distance Estimation Pipeline**: Unlike traditional LiDAR-based methods, this project achieves **highly accurate vehicle distance estimation using only image-based deep learning techniques**, making it **cost-effective and scalable**.  
✅ **Automated Dataset Preprocessing**: The project includes **scripts to automatically download, format, and prepare datasets**, reducing manual intervention and making the model easy to retrain with new data.  
✅ **High-Speed Real-Time Processing**: Optimized for **real-time performance**, the system processes video feeds with **minimal latency**, making it ideal for **autonomous vehicles and smart traffic systems**.  
✅ **Scalability for Edge AI**: With **lightweight model optimizations**, this project is deployable on **low-power edge devices**, enabling intelligent traffic monitoring solutions without requiring high-end GPUs.  

This project is a **cutting-edge contribution** to AI-driven road safety, traffic monitoring, and intelligent vehicle automation. 🚀  

---

## **Project Workflow & How to Run**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-repo/vehicle-distance-estimation.git
cd vehicle-distance-estimation
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Download & Prepare the Dataset**  
```bash
python scripts/download_datasets.py
python scripts/convert_to_yolo.py
```

### **4️⃣ Train YOLOv5 on Custom Dataset**  
```bash
python train.py --img 640 --batch 16 --epochs 50 --data config.yaml --weights yolov5s.pt --name custom_yolo
```

### **5️⃣ Perform Inference on Videos**  
```bash
python detect.py --weights runs/train/custom_yolo/weights/best.pt --source sample_video.mp4 --save-txt
```

### **6️⃣ Distance Estimation & Visualization**  
```bash
python distance_estimation.py --video input_video.mp4 --model best_model.pth
```

### **7️⃣ Generate & Save Results**  
```bash
python save_results.py --output results.csv
```

---
