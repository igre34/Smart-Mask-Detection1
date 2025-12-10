import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import config

def detector(frame, face, mask):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face.setInput(blob)
    detections = face.forward()

    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > config.CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")

            (startx, starty) = (max(0, startx), max(0, starty))
            (endx, endY) = (min(w - 1, endx), min(h - 1, endy))

            face_img = frame[starty:endY, startx:endx]
            
            # حماية من الوجوه الفارغة
            if face_img.shape[0] < 1 or face_img.shape[1] < 1:
                continue

            face_p = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_p = cv2.resize(face_p, (224, 224))
            face_p = img_to_array(face_p)
            face_p = preprocess_input(face_p)
            face_p = np.expand_dims(face_p, axis=0)

            preds = mask(face_p, training=False)
            (mask_res, withoutMask) = preds.numpy()[0]
                
            label = "Mask" if mask_res > withoutMask else "No Mask"
            prob = max(mask_res, withoutMask)
                
            results.append(((startx, starty, endx, endY), label, prob, face_img))

    # ---------------------------------------------------------
    # ⚠️ هذا السطر هو سبب المشكلة، لازم يكون راجع لليسار للنهاية
    # بمحاذاة كلمة def اللي فوق، مو داخل الـ for
    # ---------------------------------------------------------
    return results