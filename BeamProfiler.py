import sys, time, cv
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
from scipy import ndimage
from PIL import Image      
import numpy as np         
import matplotlib.pyplot as plt 
from matplotlib.pylab import *
from matplotlib.colors import LogNorm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35)
matplotlib.rc('axes', labelsize=35)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
date = os.system('date +%Y%m%d')
file_data = open(str(date),'w')

class QGui(QtGui.QWidget):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)

		self.vBox = QVBoxLayout()
		self.hBoxB = QHBoxLayout()
		self.hBoxP = QHBoxLayout()
		self.hBoxR = QHBoxLayout()
				
		self.button = QtGui.QPushButton('&Capture')
		self.button.clicked.connect(self.capture)
		self.edit1 = QLineEdit()
		self.edit1.resize(100,25)
		self.edit1.setFont(QFont("Arial",11))
		self.labelfname = QtGui.QLabel('Distance:')
			
		self.hBoxB.addWidget(self.button)			
		self.hBoxB.addWidget(self.labelfname)
		self.hBoxB.addWidget(self.edit1)
		self.hBoxB.addWidget(self.edit1)
		self.hBoxB.addWidget(self.edit1)
		
		self.scene = QtGui.QGraphicsScene(self)
		#self.scene.addText("Hello, world!");
		self.profile = QtGui.QGraphicsView(self.scene)
		self.profile.setFixedWidth(322)
		self.profile.setFixedHeight(242)

		self.fit3d = QtGui.QLabel()
		self.fit3d.setFixedWidth(322)
		self.fit3d.setFixedHeight(242)
		self.fit3d.setScaledContents(True)
			
		self.hBoxP.addWidget(self.profile)		
		self.hBoxP.addWidget(self.fit3d)			
		
		self.fitx = QtGui.QLabel()
		self.fitx.setFixedWidth(322)
		self.fitx.setFixedHeight(242)
		self.fitx.setScaledContents(True)
		
		self.fity = QtGui.QLabel()
		self.fity.setFixedWidth(322)
		self.fity.setFixedHeight(242)
		self.fity.setScaledContents(True)
		
		self.hBoxR.addWidget(self.fitx)			
		self.hBoxR.addWidget(self.fity)			
		
		self.vBox.addLayout(self.hBoxP)
		self.vBox.addLayout(self.hBoxR)
		self.vBox.addLayout(self.hBoxB)

		self.setLayout(self.vBox)
			
	def capture(self):
		print "Capturing..."
		self.name = self.edit1.text()
		#os.system(str('sh Capture.sh '+self.name))
		analice(self.name)    
		self.pixCam = QPixmap.grabWidget(self.profile)
		self.pixCam.save(str("./Fitting/profile.jpg"));
		
		self.fit3d.setPixmap(QtGui.QPixmap(str("./Fitting/3dfit.jpg")))
		#self.fit3d.setPixmap(QtGui.QPixmap(str("./Fitting/3dfit"+self.name+".jpg")))
		self.fit3d.update()
		self.fitx.setPixmap(QtGui.QPixmap(str("./Fitting/xfit.jpg")))
		#self.fitx.setPixmap(QtGui.QPixmap(str("./Fitting/xfit"+self.name+".jpg")))
		self.fitx.update()
		self.fity.setPixmap(QtGui.QPixmap(str("./Fitting/yfit.jpg")))
		#self.fity.setPixmap(QtGui.QPixmap(str("./Fitting/yfit"+self.name+".jpg")))
		self.fity.update()
		
	def get_fname(self):
		self.name=self.edit1.text()
		return self.name
		
	def ProcessFrame(self, im):
#		print "Frame update", im
		pix = QtGui.QPixmap(im)
		self.scene.clear()
		self.scene.addPixmap(pix)
		
class CamWorker(QtCore.QThread): 
    def __init__(self): 
		super(CamWorker, self).__init__() 
		self.cap = cv.CaptureFromCAM(-1)
		capture_size = (640/2,480/2)
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

    def run(self):
		while 1:
			time.sleep(0.01)
			frame = cv.QueryFrame(self.cap)
			im = QtGui.QImage(frame.tostring(), frame.width, frame.height, QtGui.QImage.Format_RGB888).rgbSwapped()
			self.emit(QtCore.SIGNAL('webcam_frame(QImage)'), im)

def analice(name):
	print "Running analysis..."	
	I = Image.open(str("./Capture/"+name+".jpg"))
	I1=I.convert('L') # convierte a escala de grises
	#I1.show()
	a=np.asarray(I1,dtype=np.uint8)/255.0 #convierte I1 en una matriz normalizada con /255
	N=a.shape[0]
	M=a.shape[1]
	max=np.max(a)
	x=np.arange(M)
	y=np.arange(N)
	X,Y = meshgrid(x,y)
   
	A=[]
	B=[]
	for i in range(N):
	    for j in range(M):
	        A.append([i,j,a[i][j]])
	        if a[i][j] ==max:
	              B.append([i,j,a[i][j]])
        
	#Image(plt)
	#print len(t)
	B = np.asarray(B)
	A = np.asarray(A)
	#sacar puntos max y min en x y y
	
	xm=int(abs(np.max (B[:,0])-np.min(B[:,0]))/2)+np.min(B[:,0])
	ym=int(abs(np.max (B[:,1])-np.min(B[:,1]))/2)+np.min(B[:,1])
	C=[]
	D=[]
	for i in range(N*M):
	    if A[i,0]==xm:
	        C.append([A[i,1],A[i,2]])
	    if A[i,1]==ym:
	        D.append([A[i,0],A[i,2]])            
	
	C=np.asarray(C)  
	D=np.asarray(D)  

	#Fit de gaussiana
	def func(x, Af, Bf, Cf):
	    return Af * np.exp(-(x-Bf)**2/(2*Cf**2)) 
	
	(Af, Bf, Cf), _ = curve_fit(func, D[:,0], D[:,1])
	fig = figure(figsize=(12,10))
	ax1 = fig.gca()
	plt.plot(D[:,0],D[:,1],'.')
	plt.plot(D[:,0],func(D[:,0], Af, Bf, Cf),linewidth = 5)
	ax1.set_xlabel("x (px)")
	ax1.set_ylabel("Intensidad")
	#print(Af, Bf, Cf)
	Ax=Af
	Bx=Bf
	Cx=Cf
	fig.set_size_inches(18.5, 10.5)
	fig.savefig('./Fitting/xfit.jpg') 
	
	(Af, Bf, Cf), _ = curve_fit(func, C[:,0], C[:,1])
	fig = figure(figsize=(12,10))
	ax2 = fig.gca()
	#ax1 = fig.add_subplot(222)
	plt.plot(C[:,0],C[:,1],'.')
	plt.plot(C[:,0],func(C[:,0], Af, Bf, Cf),linewidth = 5)
	ax2.set_xlabel("y (px)")
	ax2.set_ylabel("Intensidad")
	#print(Af, Bf, Cf)
	Ay=Af
	By=Bf
	Cy=Cf
	fig.set_size_inches(18.5, 10.5)
	fig.savefig('./Fitting/yfit.jpg') 
	
	fig=figure(figsize = (20,20))
	ax = fig.gca(projection='3d')
	figura = ax.plot_surface(X,Y,a,cmap = cm.jet,linewidth = 0)
	fig.colorbar(figura, shrink=0.5, aspect=5)  
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("Intensidad")
	ax.view_init(elev=45, azim=45)
	cset = ax.contour(X, Y, a, zdir='x', offset=-20, cmap=cm.jet)
	cset = ax.contour(X, Y, a, zdir='y', offset=-20, cmap=cm.jet)
	fig.set_size_inches(18.5, 10.5)
	fig.savefig('./Fitting/3dfit.jpg') 
	#colorbar()
	#show()
	eqx = "I="+str(Ax)+"exp(-(x-"+str(Bx)+")^2/(2"+str(Cx)+"^2))"
	eqy = "I="+str(Ay)+"exp(-(x-"+str(By)+")^2/(2"+str(Cy)+"^2))"
	file_data.write(str(Ax)+" "+str(Bx)+" "+str(Cx)+" "+str(Ay)+" "+str(By)+" "+str(Cy))

	print "Finish"

					
def main():
	a = QApplication(sys.argv)    
	a.setWindowIcon(QtGui.QIcon('./Icon/BP_Icon.png'))
	w = QWidget()
	w.resize(670, 540)
	w.setWindowTitle("Beam Profiler") 
	
	WM = QGui(w)
	WM.show()
	
	camWorker = CamWorker()
	QtCore.QObject.connect(camWorker, QtCore.SIGNAL("webcam_frame(QImage)"), WM.ProcessFrame)
	camWorker.start() 
	
	w.show()
	sys.exit(a.exec_())			
	
    
if __name__=='__main__':
	main()
