import cv2
import time

# HOG tanımlayıcısı
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Yüz tanıma için bir sınıflandırıcı
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# İlk kareyi al
ret, frame1 = cap.read()

# Hareket tespiti için ön hazırlık
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Zamanlayıcı için başlangıç zamanını al
start_time = time.time()
detected_faces = 0  # Tanınan yüz sayısı
face_tolerance = 3  # Tolerans

while True:
    # Kameradan bir kare al
    ret, frame2 = cap.read()
    
    # Hareket tespiti için mevcut kareyi işle
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    frame_diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # İnsanları tespit et
    (rects, weights) = hog.detectMultiScale(frame2, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Hareket tespiti
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Yüz tespiti
    if len(faces) > 0:
        detected_faces = 1
    
    # İnsan tespiti
    for (x, y, w, h) in rects:
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Ekranda göster
    cv2.imshow('frame', frame2)
    
    # Belirli bir süre geçtikten sonra insan sayısını ekrana yazdır
    if time.time() - start_time >= 3:
        detected_people = max(len(rects), detected_faces)
        print("Odadaki insan sayısı:", detected_people)
        start_time = time.time()
    
    # Çıkış için 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Mevcut kareyi bir sonraki kare için güncelle
    gray1 = gray2.copy()

# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()

