import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def parmak_sayisi(lm_list):
    """
    Parmak sayısını hesaplayan fonksiyon.
    Parmaklar kapalıysa, sayıyı 0 olarak döndürür.
    """
    fingers = []

    # Başparmak kontrolü (baş parmak, avuç içinden daha yukarıda mı?)
    if lm_list[4][2] < lm_list[3][2]:  # Başparmak uç kısmı avuç içinden daha yukarıda
        fingers.append(1)
    else:
        fingers.append(0)

    # Diğer parmaklar
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        if lm_list[tip][2] < lm_list[tip - 2][2]:  # Parmağın uç kısmı avuç içinden daha yukarıda mı?
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)  # Toplam parmak sayısını döndürür

while True:
    success, img = cap.read()
    if not success:
        break

    # Görüntüyü ayna gibi çeviriyoruz
    img = cv2.flip(img, 1)

    # Görüntüyü RGB'ye çeviriyoruz
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    toplam_parmak = 0

    # Eğer eller tespit edilirse
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lm_list = []
            h, w, c = img.shape

            # Elin her bir noktasını (landmarks) listeye ekliyoruz
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Parmak sayısını hesaplıyoruz
            el_toplam = parmak_sayisi(lm_list)
            toplam_parmak += el_toplam  # Toplam parmak sayısına ekliyoruz

            # Elin hangi el olduğunu belirliyoruz (sağ mı sol mu)
            hand_label = handedness.classification[0].label
            if hand_label == "Right":
                dogru_el = "Sag El"
            else:
                dogru_el = "Sol El"

            # Yazı yaz (sol veya sağ el, parmak sayısı ile birlikte)
            cv2.putText(img, f"{dogru_el}: {el_toplam} Parmak",
                        (10, 50 if dogru_el == "Sag El" else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Toplam parmak sayısı
    cv2.putText(img, f"TOPLAM: {toplam_parmak}", (5, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow("Iki El Parmak Sayisi", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()