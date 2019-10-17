import cv2   
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import glob
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir', 'data/', 'path to data')
tf.flags.DEFINE_string('out_dir', 'output/', 'path to out')

def measure_distance(pathFile):
        d=0.1
        centers=[]

        path, name = os.path.split(pathFile)
        nameP = name.rsplit('.')
        
        #img = cv2.imread('/home/oliver/Documentos/UFF-Mestrado/2019-B/AI/project3.jpeg')
        img = cv2.imread(pathFile)
        img2 = cv2.imread(pathFile)

        #img = img[250:750, 122:768]
        #img2 = img2[250:750, 122:768]
        imgE = img[250:750, 122:627]
        ###########################
        #img = img[y:y+n, x:x+m]
        img = img[455:522, 122:627]
        img2 = img2[455:522, 122:627]
        #cv2.imshow("croped result",img)

        print(img.shape)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(FLAGS.out_dir + nameP[0]+"gray.jpg",gray)

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #Se obtiene un histograma basada en las saturaciones de colores.

        ##################################################
        print("img shape ",img2.shape)
        img2[:,:,2] = cv2.equalizeHist(img2[:,:,2])
        img2[:,:,1] = cv2.equalizeHist(img2[:,:,1])
        img2[:,:,0] = cv2.equalizeHist(img2[:,:,0])

        cv2.imshow(pathFile+"entrada",img)
        cv2.imwrite(FLAGS.out_dir+ nameP[0] + "entrada.jpg",img)

        cv2.waitKey(0)

        cv2.imshow(pathFile+"imagenEQU",img2)
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"imagenEQU.jpg",img2)
        cv2.waitKey(0)

        hsv_equ = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
        #histogram = cv2.calcHist
        #print(hsv)

        print ("hsv shape:",hsv.shape)
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"imagenH.jpg", hsv[:,:,0])
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"imagenS.jpg", hsv[:,:,1])
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"imagenV.jpg", hsv[:,:,2])

        #lt.hist(img[:,:,0].ravel(),256,[0,256]); plt.show()

        color = ('b','g','r')
        hist=[]
        plt.subplot(2, 1, 1)
        for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                hist.append(histr)
                #print("hist> ",np.ravel(histr))
                plt.plot(np.ravel(histr),color = col)
                plt.title('Histograma Normal')
                plt.xlim([0,256])
        #print("ssss")
        #plt.show()


        color = ('b','g','r')
        hist=[]
        plt.subplot(2, 1, 2)
        for i,col in enumerate(color):
                histr = cv2.calcHist([img2],[i],None,[256],[0,256])
                hist.append(histr)
                #print("hist> ",np.ravel(histr))
                plt.plot(np.ravel(histr),color = col)
                plt.title('Histograma com equalizcao')
                plt.xlim([0,256])
        #print("ssss")
        ######NO PINTO PORQUE HAY ERROR
        #plt.show()


        #cv2.namedWindow('Measure DIstance',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Measure Distance', 640,480)

        #blue_lower=np.array([100,150,0],np.uint8)
        #blue_upper=np.array([140,255,255],np.uint8)
        blue_lower=np.array([80,50,0],np.uint8)
        blue_upper=np.array([150,255,255],np.uint8)

        #blue_lower=np.array([80,150,100],np.uint8)#AZUL
        #blue_upper=np.array([150,255,255],np.uint8)#AZUL


        blue=cv2.inRange(hsv,blue_lower,blue_upper) #Se crea una mascara utilizando intervalos de color azul.
        mask=cv2.inRange(hsv,blue_lower,blue_upper)

        cv2.imshow("mask of blue", blue)

        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"mask.jpg",blue)

        kernal = np.ones((3 ,3), "uint8") #Crea una matriz de 5x5 la cual recorrera el video,

        blue=cv2.erode(blue,kernal, iterations=1) #Se erosiona utilizando el kernel sobre la mascara.

        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"kernel.jpg",blue)
        cv2.imshow("kernel", blue)
        res1=cv2.bitwise_and(img, img, mask = blue) #La nueva imagen reemplazara a blue.
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"objetos.jpg",res1)
        cv2.imshow("objeto reemplazo", res1)

        (_,contours,hierarchy)=cv2.findContours(blue, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Encuentra los contornos de los objetos que se ven en el filtro

        #cv2.imwrite("contornos.jpg", contours)

        # 6.8cm (1 object) -> 2.67 inch
        #### 6.6cm (2 object) -> 2.59 inch
        ### medidad de ancho ESTIRADO
        # 8.5 -> 3.34 inch
        #yy=3.93
        yy = 0.275591
        #yy=1.92913
        #1.92913

        l_width = []
        l_dis = []
        centers = []
        #print ("contornos: ", contours)
        #cv2.drawContours(img, contours, -1, (0,255,0), 1)

        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour) #funcion de opencv que obtiene los contornos
                print ("area: ", area)
                if(area>300):
                        x,y,w,h = cv2.boundingRect(contour) #Encuentra coordenadas de los contornos.
                        #img = img[y:y+n, x:x+m]
                        #imgE = img[250:750, 122:768]
                        #img = img[455:522, 122:627]
                        imgE = cv2.rectangle(imgE,(x,y+205),(x+w,y+h+205),(0,255,0),1)
                        #res1 = cv2.rectangle(imgE,(x,y),(x+w,y+h),(0,255,0),1)

                        print("pixel width: ", w)
                        l_width.append(w+7)
                        cv2.putText(imgE,"Objeto",(x,y-10+205),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))


                        M = cv2.moments(contour) #Se obtiene el centro de masa de los marcadores enconrados.
                        cx = int(M['m10'] /M['m00'])
                        cy = int(M['m01'] /M['m00'])
                        centers.append((cx,cy))
                        print("center: ", centers)
                        cv2.circle(imgE, (cx, cy+205), 2, (0, 0, 255), -1)
                        cv2.circle(res1, (cx, cy), 1, (0, 0, 255), -1)
                        
                ####
                if len(centers)==2:
                        res1 = cv2.line(res1, (centers[0][0], centers[0][1]), (centers[1][0], centers[1][1]), (0, 255, 0), 1)
                        print("cx: ",cx)
                        print("cy: ", cy)
                        #D = np.linalg.norm(centers,cx-cy) #Se aplica distancia euclidiana para encontrar la distancia entre los centros de masa.
                        D = math.sqrt(math.pow((centers[1][0] - centers[0][0]) , 2) + math.pow((centers[1][1] - centers[0][1] ) , 2) )
                        l_dis.append(D)
                        print("dist pixel: ",D)

        #print (centers)
        #euclidea = math.sqrt(math.pow(centers[1][0] - centers[0][0] ,2) + math.pow(centers[1][1] - centers[1][0] ,2))
        #print ("euc ",euclidea)

        #l_dis.append(euclidea)

        #pixel_per_metric = (l_dis[0]) / float(yy)
        #pixel_per_metric = 150.6502502485687
        pixel_per_metric = (l_width[0]) / float(yy)

        print ("pixelmetric: ",pixel_per_metric)

        dist = l_dis[0] / float(pixel_per_metric )

        print("distancia: ", dist)
        cv2.putText(imgE,"Distancia = "+str(round(dist*2.54, 2))+" cm", (20,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
        
        #cv2.namedWindow('Measure DIstance',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Measure Distance',640,480)
        cv2.imshow("Measure Distance",imgE)
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"distance.jpg",imgE)
        cv2.imshow("distance2",res1)
        cv2.imwrite(FLAGS.out_dir+ nameP[0] +"distanceObject.jpg",res1)
        cv2.waitKey(0)
        # (nome, pixel, pixel_uni, inch, cm )
        return (nameP[0] ,round(l_dis[0],3), centers[0][0] - centers[1][0] ,round(dist,3), round(dist*2.54, 3))

if __name__ == '__main__':
        
        print("data ", FLAGS.data_dir)
        print ("out ", FLAGS.out_dir)
        list_distancias = []
        files = sorted(glob.glob(FLAGS.data_dir+'/*.png'))
        for x in files:
                list_distancias.append(measure_distance(x))
        ##############################
        print("###########################")
        print("Distancias:")
        for e in list_distancias:
                print(str(e[0])+"  \t" +str(e[1])+"px  \t" + str(e[2])+"uni px  \t" +str(e[3])+"inch  \t" +str(e[4])+"cm \n")
        print("Deformacion Promedio:")
        ini = list_distancias[0]
        fin = list_distancias[len(list_distancias)-1]
        siz = len(list_distancias)-1
        print(str((fin[1]-ini[1])/siz)+"px\t" +str((fin[2]-ini[2])/siz)+"inch\t"+str((fin[3]-ini[3])/siz)+"cm \n")
        