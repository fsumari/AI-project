import cv2   
import numpy as np

#Captura de video a traves de la webcam
#cap=cv2.VideoCapture(0)

#while(1):
d=0.1
centers=[]
#_, img = cap.read()

#img = cv2.imread('/home/oliver/Documentos/UFF-Mestrado/2019-B/AI/project3.jpeg')
img = cv2.imread('/home/oliver/Documentos/2019B/AI/estirada3.jpeg')

print(img.shape)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #Se obtiene un histograma basada en las saturaciones de colores.

#cv2.namedWindow('Measure DIstance',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Measure Distance', 640,480)

#blue_lower=np.array([80,150,100],np.uint8)
#blue_upper=np.array([150,255,255],np.uint8)
blue_lower=np.array([38,25,25],np.uint8)
blue_upper=np.array([86,255,255],np.uint8)

blue=cv2.inRange(hsv,blue_lower,blue_upper) #Se crea una mascara utilizando intervalos de color azul.

kernal = np.ones((5 ,5), "uint8") #Crea una matriz de 5x5 la cual recorrera el video,

blue=cv2.erode(blue,kernal, iterations=1) #Se erosiona utilizando el kernel sobre la mascara.
res1=cv2.bitwise_and(img, img, mask = blue) #La nueva imagen reemplazara a blue.


(_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encuentra los contornos de los objetos que se ven en el filtro

# 6.8cm (1 object) -> 2.67 inch
#### 6.6cm (2 object) -> 2.59 inch
### medidad de ancho ESTIRADO
# 8.5 -> 3.34 inch
#yy=2.67
yy=3.34

l_width = []
l_dis = []
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour) #funcion de opencv que obtiene los contornos
    if(area>300):
        x,y,w,h = cv2.boundingRect(contour) #Encuentra coordenadas de los contornos.
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        print("pixel width: ", w )
        l_width.append(w)
        cv2.putText(img,"Objeto",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))


        M = cv2.moments(contour) #Se obtiene el centro de masa de los marcadores enconrados.
        cx = int(M['m10'] /M['m00'])
        cy = int(M['m01'] /M['m00'])
        centers.append([cx,cy])
        cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)

    if len(centers)==2:
        D = np.linalg.norm(cx-cy) #Se aplica distancia euclidiana para encontrar la distancia entre los centros de masa.
        l_dis.append(D)
        print(D)

pixel_per_metric = l_width[0] / float(yy)
print ("pixelmetric: ",pixel_per_metric)
dist = l_dis[0] / float(pixel_per_metric )
print("distancia: ", dist)
cv2.putText(img,"Distancia = "+str(round(dist*2.54, 2))+" cm", (20,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

#cv2.namedWindow('Measure DIstance',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Measure Distance',640,480)
cv2.imshow("Measure Distance",img)
cv2.waitKey(0)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    cap.release()
    #    cv2.destroyAllWindows()
    #    break
