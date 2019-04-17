import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_hist(frame, frame1,k=0,l=0,frame_name='frame', frame1_name='frame1'):
    #ヒストグラムの表示
    ax[k,l].plot(frame,label=str(frame_name))
    ax[k,l].plot(frame1,label=str(frame1_name))
    ax[k,l].set_xlim([0, 256])
    ax[k,l].set_ylim([0,max(max(frame),max(frame1))])  #15000])
    ax[k,l].legend()
    #plt.show()

def plot_hist1(frame, k=0,frame_name='frame'):
    #ヒストグラムの表示
    ax[k].plot(frame,label=str(frame_name))
    ax[k].set_xlim([0, 256])
    ax[k].set_ylim([0,max(frame)])  #15000])
    ax[k].legend()
    #plt.show()
    
def something(frame,YCrCb=0):
    imgOrg = frame  #cv2.imread('61.jpg', 1)

    #BGRをYCrCbに変換します
    orgYCrCb = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2YCR_CB)

    #輝度のヒストグラムを作成
    histOrgY = cv2.calcHist([orgYCrCb], [YCrCb], None, [256], [0, 256]) #0:Y 1:Cr 2:Cb
    return histOrgY

def something2(frame, frame1,YCrCb=0):
    imgOrg = frame  #cv2.imread('61.jpg', 1)
    imgLut = frame1  #cv2.imread('LUT.jpg', 1)

    #BGRをYCrCbに変換します
    orgYCrCb = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2YCR_CB)
    lutYCrCb = cv2.cvtColor(imgLut, cv2.COLOR_BGR2YCR_CB)

       
    #輝度のヒストグラムを作成
    histOrgY = cv2.calcHist([orgYCrCb], [YCrCb], None, [256], [0, 256]) #0:Y 1:Cr 2:Cb
    histLutY = cv2.calcHist([lutYCrCb], [YCrCb], None, [256], [0, 256])
    return histOrgY,histLutY

#fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
img = cv2.imread('s1.jpg',1)
img5 = cv2.imread('s7.jpg',1)
plt.imshow(img5)
plt.title("img5")
cv2.imwrite('img5.jpg', img5)
plt.show()
plt.imshow(img)
plt.title("img")
cv2.imwrite('img.jpg', img)
plt.show()

lab = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
img1 = cv2.split(lab)
list =('l','a','b')
for s in range(0,1):
    hist,bins = np.histogram(img1[s].flatten(),256,[0,100])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    lst = str(list[s])
    ax1[s].plot(cdf_normalized, color = 'b',label='cdf_'+str(lst))
    ax1[s].hist(img1[s].flatten(),256,[0,256], color = 'r',label='histogram_'+str(lst))
    ax1[s].set_xlim([0,256])
    ax1[s].legend(loc = 'upper left')
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
 
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img1[s] = cdf[img1[s]]
plt.show()

img1[1]=img1[1]
img1[2]=img1[2]
img1 = cv2.merge(img1)
img2 = cv2.cvtColor(img1, cv2.COLOR_YCrCb2BGR)
plt.imshow(img2)
plt.title("img2")
cv2.imwrite('img2.jpg', img2)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
histOrgY=something(img2,0)
plot_hist1(histOrgY,0, "img2")
histOrgY=something(img2,1)
plot_hist1(histOrgY,1, "img2")
histOrgY=something(img2,2)
plot_hist1(histOrgY,2,"img2")
#cv2.imwrite('img2_hist.jpg', ax)
plt.show()

img3 = cv2.split(img)
list =('r','g','b')
hist_r = np.bincount(img3[0].ravel(),minlength=256)
hist_g = np.bincount(img3[1].ravel(),minlength=256)
hist_b = np.bincount(img3[2].ravel(),minlength=256)
# グラフの作成
plt.xlim(0, 255)
plt.plot(hist_r, "-r", label="Red")
plt.plot(hist_g, "-g", label="Green")
plt.plot(hist_b, "-b", label="Blue")
plt.xlabel("Pixel value", fontsize=20)
plt.ylabel("Number of pixels", fontsize=20)
plt.legend()
plt.grid()
plt.show()

for s in range(0,3):
    if s==0:
        hist,bins = np.histogram(img3[s].flatten(),256,[0,125])
    else:    
        hist,bins = np.histogram(img3[s].flatten(),256,[0,155])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    lst = str(list[s])
    ax1[s].plot(cdf_normalized, color = 'b',label='cdf_'+str(lst))
    ax1[s].hist(img3[s].flatten(),256,[0,256], color = 'r',label='histogram_'+str(lst))
    ax1[s].set_xlim([0,256])
    ax1[s].legend(loc = 'upper left')
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    x = np.linspace(0, 255, 256)    
    plt.plot(x,cdf)
    plt.title('cdf_'+str(s)+'.jpg')
    plt.pause(0.1)
    plt.savefig('cdf_'+str(s)+'.jpg')
    plt.close()
    img3[s] = cdf[img3[s]]
 
img4 = cv2.merge(img3)

lab = cv2.cvtColor(img4, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
img6 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

plt.imshow(img6)
plt.title("img6")
cv2.imwrite('img6.jpg', img6)
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 4))
histOrgY,histLutY=something2(img5,img6,0)
plot_hist(histOrgY,histLutY,0,0,"img5", "img6")
histOrgY,histLutY=something2(img5,img6,1)
plot_hist(histOrgY,histLutY,0,1,"img5", "img6")
histOrgY,histLutY=something2(img5,img6,2)
plot_hist(histOrgY,histLutY,0,2,"img5","img6")
#cv2.imwrite('img4_hist.jpg',ax)
plt.show()