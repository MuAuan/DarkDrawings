import cv2
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.8
img_src = cv2.imread('s1.jpg', 1)

for gm in range(10,50,2):
    gamma=gm/10
    lookUpTable = np.zeros((256, 1), dtype = 'uint8')

    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    
    # ルックアップテーブルによるガンマ補正
    img_gamma = cv2.LUT(img_src, lookUpTable)

    plt.imshow(img_gamma)
    plt.title('gamma_'+str(gamma)+'.jpg')
    cv2.imwrite('gamma_'+str(gamma)+'.jpg', img_gamma)
    plt.pause(0.1)
    plt.close()
    
    x = np.linspace(0, 255, 256)    
    plt.plot(x,lookUpTable)
    plt.title('lookUpTable_'+str(gamma)+'.jpg')
    plt.pause(0.1)
    plt.savefig('lookUpTable_'+str(gamma)+'.jpg')
    plt.close()

