import os

import cv2

# Input画像とOutput画像の名前を入力
inimgname = "image.jpg"
outimgname = "beauty_" + inimgname
# 各パーツのリサイズ比を入力
eyex = 1.1
eyey = 1.3
nosex = 0.8
nosey = 1.1
mouthx = 1.2
mouthy = 1.1

# 1. ファイルからの画像取得
img = cv2.imread(inimgname)
cv2.imshow("input", img)
cv2.waitKey(1)
# cv2.destroyAllWindows()

# 2. 画像中の目・鼻・口を囲む矩形領域の抽出

# カスケード分類器の準備
facecc_xml = "haarcascades/haarcascade_frontalface_alt2.xml"  # cc<-cascade
lefteyecc_xml = "haarcascades/haarcascade_mcs_lefteye_alt.xml"
righteyecc_xml = "haarcascades/haarcascade_mcs_righteye_alt.xml"
nosecc_xml = "haarcascades/haarcascade_mcs_nose.xml"
mouthcc_xml = "haarcascades/haarcascade_mcs_mouth.xml"
facecc = cv2.CascadeClassifier(facecc_xml)
lefteyecc = cv2.CascadeClassifier(lefteyecc_xml)
righteyecc = cv2.CascadeClassifier(righteyecc_xml)
nosecc = cv2.CascadeClassifier(nosecc_xml)
mouthcc = cv2.CascadeClassifier(mouthcc_xml)

# 顔の検出
facerects = facecc.detectMultiScale(img)  # rect<-rectangle
print(facecc)

for facerect in facerects:
    (x, y, w, h) = facerect
    face = img[y:y + h, x:x + w]
    #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

    # 左目の検出
    leftupface = face[:int(len(face) / 2), int(len(face[0]) / 2):]
    lefteyerects = lefteyecc.detectMultiScale(leftupface)
    for lefteyerect in lefteyerects:
        (lex, ley, lew, leh) = lefteyerect
    # img = cv2.rectangle(img,(x+int(len(face[0])/2)+lex,y+ley),(x+int(len(face[0])/2)+lex+lew,y+ley+leh),(0,0,255),1)

    # 右目の検出
    rightupface = face[:int(len(face) / 2), :int(len(face[0]) / 2)]
    righteyerects = righteyecc.detectMultiScale(rightupface)
    for righteyerect in righteyerects:
        (rex, rey, rew, reh) = righteyerect
    #         img = cv2.rectangle(img,(x+rex,y+rey),(x+rex+rew,y+rey+reh),(0,0,255),1)

    # 鼻の検出
    middleface = face[int(len(face) / 4):int(len(face) * 3 / 4), int(len(face) / 4):int(len(face) * 3 / 4)]
    noserects = nosecc.detectMultiScale(middleface)
    for noserect in noserects:
        (nx, ny, nw, nh) = noserect
    # img = cv2.rectangle(img,(int(len(face)/4)+x+nx,int(len(face)/4)+y+ny),(int(len(face)/4)+x+nx+nw,
    # int(len(face)/4)+y+ny+nh),(0,0,255),1)

    # 口の検出
    bottomface = face[int(len(face) / 2):, int(len(face) / 4):int(len(face) * 3 / 4)]
    mouthrects = mouthcc.detectMultiScale(bottomface)
    for mouthrect in mouthrects:
        (mx, my, mw, mh) = mouthrect
    # img = cv2.rectangle(img,(int(len(face)/4)+x+mx,int(len(face)/2)+y+my),(int(len(face)/4)+x+mx+mw,
    # int(len(face)/2)+y+my+mh),(0,0,255),1)

    # 3. 各矩形領域に対するサイズ調整

    # 左目を大きく
    for lefteyerect in lefteyerects:
        (lex, ley, lew, leh) = lefteyerect
        lefteye = leftupface[ley:ley + leh, lex:lex + lew]
        biglefteye = cv2.resize(lefteye, None, fx=eyex, fy=eyey)
        # 4. サイズ調整後の各矩形領域の重ね合わせ
        starty = int(y + ley + len(lefteye) / 2 - len(biglefteye) / 2)
        endy = int(y + ley + len(lefteye) / 2 + len(biglefteye) / 2)
        startx = int(x + len(face[0]) / 2 + lex + len(lefteye[0]) / 2 - len(biglefteye[0]) / 2)
        endx = int(x + len(face[0]) / 2 + lex + len(lefteye[0]) / 2 + len(biglefteye[0]) / 2)
        img[starty:endy, startx:endx] = biglefteye

    # 右目を大きく
    for righteyerect in righteyerects:
        (rex, rey, rew, reh) = righteyerect
        righteye = rightupface[rey:rey + reh, rex:rex + rew]
        bigrighteye = cv2.resize(righteye, None, fx=eyex, fy=eyey)
        # 4. サイズ調整後の各矩形領域の重ね合わせ
        starty = int(y + rey + len(righteye) / 2 - len(bigrighteye) / 2)
        endy = int(y + rey + len(righteye) / 2 + len(bigrighteye) / 2)
        startx = int(x + rex + len(righteye[0]) / 2 - len(bigrighteye[0]) / 2)
        endx = int(x + rex + len(righteye[0]) / 2 + len(bigrighteye[0]) / 2)
        img[starty:endy, startx:endx] = bigrighteye

    # 鼻を小さく
    for noserect in noserects:
        (nx, ny, nw, nh) = noserect
        nose = middleface[ny:ny + nh, nx:nx + nw]
        smallnose = cv2.resize(nose, None, fx=nosex, fy=nosey)
        # 4. サイズ調整後の各矩形領域の重ね合わせ
        starty = int(len(face) / 4 + y + ny + len(nose) / 2 - len(smallnose) / 2)
        endy = int(len(face) / 4 + y + ny + len(nose) / 2 + len(smallnose) / 2)
        startx = int(len(face) / 4 + x + nx + len(nose[0]) / 2 - len(smallnose[0]) / 2)
        endx = int(len(face) / 4 + x + nx + len(nose[0]) / 2 + len(smallnose[0]) / 2)
        img[starty:endy, startx:endx] = smallnose

    # 口を大きく
    for mouthrect in mouthrects:
        (mx, my, mw, mh) = mouthrect
        mouth = bottomface[my:my + mh, mx:mx + mw]
        bigmouth = cv2.resize(mouth, None, fx=mouthx, fy=mouthy)
        # 4. サイズ調整後の各矩形領域の重ね合わせ
        starty = int(len(face) / 2 + y + my + len(mouth) / 2 - len(bigmouth) / 2)
        endy = int(len(face) / 2 + y + my + len(mouth) / 2 + len(bigmouth) / 2)
        startx = int(len(face) / 4 + x + mx + len(mouth[0]) / 2 - len(bigmouth[0]) / 2)
        endx = int(len(face) / 4 + x + mx + len(mouth[0]) / 2 + len(bigmouth[0]) / 2)
        img[starty:endy, startx:endx] = bigmouth

    cv2.imshow("output", img)
    cv2.waitKey(100)
cv2.imwrite(outimgname, img)
cv2.destroyAllWindows()
