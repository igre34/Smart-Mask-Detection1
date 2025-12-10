import cv2
import time
import config           
import ai_loader        
import detector         
def main():
    face_model, mask_model = ai_loader.load_ai_mod()

    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2.0) 
    last_snapshot_time = 0

    while True:
       
        ret, frame = vs.read()
        
        
        if not ret:
            break

        
        frame = cv2.resize(frame, (800, 600))
        
        results = detector.detector(frame, face_model, mask_model)

        for (box, label, prob, face_img) in results:
            (startX, startY, endX, endY) = box
            
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            if label == "No Mask":
                current_time = time.time()
                if (current_time - last_snapshot_time) > config.SNAPSHOT_COOLDOWN:
                    img_name = f"VIOLATION_{int(current_time)}.jpg"
                    
                    zoomed_face = cv2.resize(face_img, (400, 400))
                    
                    cv2.imwrite(img_name, zoomed_face)
                    print(f"[ALERT] Violation Detected! Snapshot saved: {img_name}")
                    
                    last_snapshot_time = current_time

            text = "{}: {:.2f}%".format(label, prob * 100)
            cv2.putText(frame, text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Student Mask Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.release()

if __name__ == "__main__":
    main()