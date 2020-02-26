import cv2
import numpy as np

resim=cv2.imread("scofield.jpg")
    
mask=np.zeros(resim.shape[:2],np.uint8)#resim.shape(217,232,3)

bgdModel=np.zeros((1,65),dtype=np.float64)
                  
fgdModel=np.zeros((1,65),dtype=np.float64)

rect=(10,10,195,300) #BELİRLEDĞİMİZ ALANIN BOYUTLARINI BELİRTİYORUZ.(X,Y,YUKSEKLİK,EN)

cv2.grabCut(resim,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#Eğer maskk == 2 veya mask == 1 ise, mask2 0 olsun, aksi takdirde 'uint8' tipi olarak 1 alır.
# maskeye rgb için ek boyut ekleyerek, varsayılan olarak alır 1
# bölümlenmiş görüntüyü elde etmek için giriş görüntüsü ile çarpın

mask2=np.where((mask==0) | (mask==2),0,1).astype(np.uint8)#NUMPY.WHERE((KOSUL),X,Y) KOSULLARA GÖRE ELEMAN SEÇİMİ SAĞLAR
#for x in range(mask2.shape[0]):
#    for y in range(mask2.shape[1]):
#        print("Satir : {} Sütün : {} deger : {}".format(x,y,mask2[x,y]))
print(resim.shape)

print(mask2)

resim=resim*mask2[:,:,np.newaxis]#NEWAXİS DİZİYE YENİ BİR BOYUT KATMAK İÇİN KULLANILIR
cv2.imshow("Resim",resim)
cv2.waitKey(0)
cv2.destroyAllWindows()



#>>> a[:, np.newaxis]
#array([[0],
#       [1],
#       [2],
#       [3],
#       [4],
#       [5],
#       [6],
#       [7],
#       [8],
#       [9]])
#>>> a[:, np.newaxis].shape
#(10, 1)




