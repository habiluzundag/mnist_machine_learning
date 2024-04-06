import cv2
import numpy as np
from keras.models import load_model

# Eğitilmiş modelin yüklenmesi
model = load_model("my_model.h5")


# Fare olaylarını yakalamak için işlev
def draw_and_predict(event, x, y, flags, param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        img_resize = cv2.resize(img, (28, 28))
        img_resize = np.expand_dims(np.expand_dims(img_resize, axis=-1), axis=0)
        prediction = model.predict(img_resize)
        predicted_class = np.argmax(prediction)
        print("Tahmin edilen rakam:", predicted_class)


# Boş bir siyah ekran oluşturma
img = np.zeros((200, 200), dtype=np.uint8)

# Pencere oluşturma ve fare olaylarını yakalama
cv2.namedWindow("Drawing")
cv2.setMouseCallback("Drawing", draw_and_predict)

while True:
    cv2.imshow("Drawing", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
