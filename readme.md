# Drowsiness Detection using OpenCV 😴🚫🚗

This project detects eye closure in real time and alerts the user when drowsiness is detected.

---

## About This Project

This implementation is **based on an open-source project originally created by Akshay Bahadur**.  
I studied the approach and built my own working version as part of my learning in Computer Vision and Python.

This repository contains:
✔ the working detection pipeline  
✔ the model files  
✔ setup configuration  
✔ documentation created while understanding the Eye Aspect Ratio (EAR)–based drowsiness detection method  

This project helped me understand real-time video processing, facial landmark detection, and fatigue monitoring systems.

---

## Applications 🎯
This can help reduce road accidents caused by fatigue, especially for drivers travelling long distances.

---

## Code Requirements 🦄
Python 2.7+ or any Python 3.x version.

---

## Dependencies

- OpenCV  
- imutils  
- dlib  
- scipy  

---

## Description 📌

A computer-vision system that detects drowsiness in real-time webcam video and plays an alert if the user appears sleepy.  
The system calculates the **Eye Aspect Ratio (EAR)** using facial landmark detection.

If EAR remains below a threshold for multiple continuous frames, the user is considered drowsy.

---

## Algorithm 👨‍🔬

Each eye is represented by 6 key-points around the eyelid.

The EAR value is computed and monitored for 20 consecutive frames.  
If the EAR value falls below **0.25**, an alert is triggered.

---

## My Contributions ✨

- Set up and configured the project environment  
- Integrated and tested the pretrained landmark model  
- Implemented EAR-based eye-closure detection  
- Debugged and ran real-time webcam processing  
- Documented the project for clarity  

This work strengthened my understanding of:

✔ OpenCV pipelines  
✔ dlib landmark detection  
✔ EAR fatigue metrics  
✔ real-time inference performance  

---

## Results 📊

Drowsiness is successfully detected and alerts trigger when eye closure persists.

---

## How to Run ▶️


Allow webcam permission when prompted.

---

## Model File

`shape_predictor_68_face_landmarks.dat`  
is the pretrained model used for facial landmark detection.

---

## Credits ❤️

Original Author: **Akshay Bahadur**  
Repository: https://github.com/akshaybahadur21/Drowsiness_Detection  

This repo contains my modified working version created for learning and academic purposes.

---

## License 📜

This project uses the MIT License.
Original license and credit remain with the author.

---

## References 🔱
- Adrian Rosebrock — PyImageSearch Blog  
  https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
