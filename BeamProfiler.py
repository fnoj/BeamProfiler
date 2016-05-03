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
import time

matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35)
matplotlib.rc('axes', labelsize=35)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
date = time.strftime("%d%m%y%H%M")
file_data = open(str("./Data/"+date+".dat"),"w+")
data_fit=0
distance=0
FILENAME=0
Xback=0
Yback=0

class WidgetCam(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Profile")
		self.hBoxP = QHBoxLayout()
		self.scene = QtGui.QGraphicsScene(self)
		#self.scene.addText("Hello, world!");
		self.profile = QtGui.QGraphicsView(self.scene)
		self.profile.setFixedWidth(322)
		self.profile.setFixedHeight(242)
		self.hBoxP.addWidget(self.profile)
		self.setLayout(self.hBoxP)
		
	def ProcessFrame(self, im):
		#print "Frame update", im
		self.pix = QtGui.QPixmap(im)
		self.scene.clear()
		self.scene.addPixmap(self.pix)
	
	def savePhoto(self):
		self.pixMap = QPixmap.grabWidget(self.profile);
		self.pixMap.save(str("./Fitting/"+FILENAME+".jpg"))
		
class WidgetFit3d(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Profile 3D")
		self.hBoxP = QHBoxLayout()
		self.fit3d = QtGui.QLabel()
		self.fit3d.setFixedWidth(322)
		self.fit3d.setFixedHeight(242)
		self.fit3d.setScaledContents(True)
		self.hBoxP.addWidget(self.fit3d)	
		self.setLayout(self.hBoxP)
	
	def setPicture(self,namefile):
		self.fit3d.setPixmap(QtGui.QPixmap(str(namefile)))
		self.fit3d.update()
		
class WidgetFitX(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Profile X")
		self.hBoxP = QHBoxLayout()
		self.fitx = QtGui.QLabel()
		self.fitx.setFixedWidth(322)
		self.fitx.setFixedHeight(242)
		self.fitx.setScaledContents(True)
		self.hBoxP.addWidget(self.fitx)	
		self.setLayout(self.hBoxP)
		
	def setPicture(self,namefile):
		self.fitx.setPixmap(QtGui.QPixmap(str(namefile)))
		self.fitx.update()
		
class WidgetFitY(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Profile Y")
		self.hBoxP = QHBoxLayout()
		self.fity = QtGui.QLabel()
		self.fity.setFixedWidth(322)
		self.fity.setFixedHeight(242)
		self.fity.setScaledContents(True)
		self.hBoxP.addWidget(self.fity)	
		self.setLayout(self.hBoxP)
		
	def setPicture(self,namefile):
		self.fity.setPixmap(QtGui.QPixmap(str(namefile)))
		self.fity.update()
		
class WidgetWaistX(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Waist X")
		self.hBoxP = QHBoxLayout()
		self.waistx = QtGui.QLabel()
		self.waistx.setFixedWidth(322)
		self.waistx.setFixedHeight(242)
		self.waistx.setScaledContents(True)
		self.hBoxP.addWidget(self.waistx)	
		self.setLayout(self.hBoxP)
		
	def setPicture(self,namefile):
		self.waistx.setPixmap(QtGui.QPixmap(str(namefile)))
		self.waistx.update()		

class WidgetWaistY(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Waist Y")
		self.hBoxP = QHBoxLayout()
		self.waisty = QtGui.QLabel()
		self.waisty.setFixedWidth(322)
		self.waisty.setFixedHeight(242)
		self.waisty.setScaledContents(True)
		self.hBoxP.addWidget(self.waisty)	
		self.setLayout(self.hBoxP)
		
	def setPicture(self,namefile):
		self.waisty.setPixmap(QtGui.QPixmap(str(namefile)))
		self.waisty.update()		

class WidgetControl(QtGui.QDialog):
	def __init__(self,wcamo,wfitxo,wfityo,wfit3do,wdatao,parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Control")
		self.bcapture = QtGui.QPushButton('&Capture', self)
		self.bcapture.clicked.connect(self.capture)
		self.labeldistance = QtGui.QLabel()
		self.labeldistance.setText("Distance [mm]")
		self.lineEdit = QtGui.QLineEdit()
		self.lineEdit.resize(100,25)
		self.lineEdit.setFont(QFont("Arial",11))
		self.bback = QtGui.QPushButton('&Backgroud', self)
		self.bback.clicked.connect(self.back)		
		self.bwaist = QtGui.QPushButton('&Waist', self)
		self.bwaist.clicked.connect(self.fitwaist)
		layout = QtGui.QVBoxLayout(self)
		layout.addWidget(self.bcapture)
		layout.addWidget(self.labeldistance)
		layout.addWidget(self.lineEdit)
		layout.addWidget(self.bback)		
		layout.addWidget(self.bwaist)
		self.setLayout(layout)
		self.wcam=wcamo
		self.wfitx=wfitxo
		self.wfity=wfityo
		self.wfit3d=wfit3do
		self.wdata=wdatao
		self.waistx=WidgetWaistX()
		self.waisty=WidgetWaistY()
	
	def capture(self):
		global distance
		global FILENAME
		distance=self.lineEdit.text()
		FILENAME=str(str(date)+"_"+str(distance)+"_")
		print "Profile saved as ./Fitting/"+FILENAME+".jpg"
		self.wcam.savePhoto()
		analice(str("./Fitting/"+str(FILENAME)+".jpg"))
		self.wfity.setPicture(str("./Fitting/"+str(FILENAME)+"X.jpg"))
		self.wfitx.setPicture(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
		self.wfit3d.setPicture(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
		self.wdata.setValues()
	
	def back(self):
		global Xback
		global Yback
		self.wcam.savePhoto()
		I = Image.open(str("./Fitting"+FILENAME+".jpg"))
		I1=I.convert('L') # convierte a escala de grises
		a=np.asarray(I1,dtype=np.uint8)/255.0 #convierte I1 en una matriz normalizada con /255
		N=a.shape[0]
		M=a.shape[1]
		max=np.max(a)
		x=np.arange(M)
		y=np.arange(N)
		X,Y = meshgrid(x,y)
		Xback=X
		Yback=Y
		
	def fitwaist(self):
		global file_data
		global FILENAME
		file_data.close()
		fig = figure(figsize=(12,10))
		date="W11"##############
		lam=632./(10**6)
		Zd, Ax, Bx, Cx, Ay, By, Cy = loadtxt(str("./Data/"+date+".dat"), unpack = True)		
		def func(x, Af, Bf, Cf, Df):
			return Af*(np.sqrt(((x-Bf)/Cf)**2+1))+Df
		(Af, Bf, Cf, Df), _ = curve_fit(func, Zd, Ax)
		plt.plot(Zd,Ax,'.')
		plt.plot(Zd,func(Zd, Af, Bf, Cf, Df),linewidth = 2)
		zr=Cf
		Wo = np.sqrt((zr*lam)/np.pi)
		X = np.linspace(-600, 600, 1000, endpoint=True)
		plt.plot(X,func(X, Af, Bf, Cf, Df),linewidth = 1)
		plt.plot(X,func(X, -Af, Bf, Cf, Df),linewidth = 1)
		fig.savefig(str("./Fitting/"+str(FILENAME)+"WaistX.jpg"))
		self.waistx.setPicture(str("./Fitting/"+str(FILENAME)+"WaistX.jpg"))
		self.waistx.show()
		
		plt.plot(Zd,Ay,'.')
		plt.plot(Zd,func(Zd, Af, Bf, Cf, Df),linewidth = 2)
		zr=Cf
		Wo = np.sqrt((zr*lam)/np.pi)
		Y = np.linspace(-600, 600, 1000, endpoint=True)
		plt.plot(Y,func(Y, Af, Bf, Cf, Df),linewidth = 1)
		plt.plot(Y,func(Y, -Af, Bf, Cf, Df),linewidth = 1)
		fig.savefig(str("./Fitting/"+str(FILENAME)+"WaistY.jpg"))
		self.waisty.setPicture(str("./Fitting/"+str(FILENAME)+"WaistY.jpg"))
		self.waisty.show()
				
		
class WidgetData(QtGui.QDialog):
	def __init__(self, parent=None):
		super(WidgetData, self).__init__(parent)
		self.setWindowTitle("Data")
		self.layout = QtGui.QGridLayout() 
		self.table = QtGui.QTableWidget()
		self.layout.addWidget(self.table)
		self.table.setRowCount(6)
		self.table.setColumnCount(2)
		self.table.setItem(0,0, QTableWidgetItem("A_x"))
		self.table.setItem(1,0, QTableWidgetItem("B_x"))
		self.table.setItem(2,0, QTableWidgetItem("C_x"))
		self.table.setItem(3,0, QTableWidgetItem("A_y"))
		self.table.setItem(4,0, QTableWidgetItem("B_y"))
		self.table.setItem(5,0, QTableWidgetItem("C_y"))
		self.setLayout(self.layout)
	
	def setValues(self):
		self.table.setItem(0,1, QTableWidgetItem(str(data_fit[0])))
		self.table.setItem(1,1, QTableWidgetItem(str(data_fit[1])))
		self.table.setItem(2,1, QTableWidgetItem(str(data_fit[2])))
		self.table.setItem(3,1, QTableWidgetItem(str(data_fit[3])))
		self.table.setItem(4,1, QTableWidgetItem(str(data_fit[4])))
		self.table.setItem(5,1, QTableWidgetItem(str(data_fit[5])))
		self.table.update()
	
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

class WMainProfiler(QtGui.QMainWindow):
	def __init__(self):
		super(WMainProfiler, self).__init__()
		
		self.wcam = WidgetCam()
		self.wfit3d = WidgetFit3d()	
		self.wfitx = WidgetFitX()
		self.wfity = WidgetFitY()
		self.wdata=WidgetData()
		self.wcontrol=WidgetControl(self.wcam,self.wfitx,self.wfity,self.wfit3d,self.wdata)
				
		self.statusBar().showMessage('Ready')
		self.setWindowTitle('Statusbar')
		self.show()
		
		exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('Exit application')
		exitAction.triggered.connect(QtGui.qApp.quit)
	
		#captureAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Capture', self)
		#captureAction.setShortcut('Ctrl+C')
		#captureAction.setStatusTip('Capture Profile')
		#captureAction.triggered.connect(self.capture)
		
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAction)
		
		#optionsMenu = menubar.addMenu('&Options')
		#optionsMenu.addAction(captureAction)
		
		self.setWindowState(Qt.WindowMaximized)
		self.setWindowTitle('Beam Profiler')
		self.show()
		
		self.wcam.setGeometry(QRect(10, 120, 350, 200))
		camWorker = CamWorker()
		QtCore.QObject.connect(camWorker, QtCore.SIGNAL("webcam_frame(QImage)"), self.wcam.ProcessFrame)
		camWorker.start()
		self.wcam.show()
		
		self.wfit3d.setGeometry(QRect(380, 120, 350, 200))
		self.wfit3d.show()
		
		self.wfitx.setGeometry(QRect(10, 420, 350, 200))
		self.wfitx.show()
		
		self.wfity.setGeometry(QRect(380, 420, 350, 200))
		self.wfity.show()
		
		self.wdata.setGeometry(QRect(880, 120, 239, 225))
		self.wdata.show()
		
		self.wcontrol.setGeometry(QRect(880, 420, 100, 200))
		self.wcontrol.show()		
		
	#def capture(self):
		#self.wcam.savePhoto()
		#analice(str("./Fitting/"+str(FILENAME)+".jpg"))
		#self.wfity.setPicture(str("./Fitting/"+str(FILENAME)+"X.jpg"))
		#self.wfitx.setPicture(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
		#self.wfit3d.setPicture(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
		#self.wdata.setValues()
		
def analice(fname):
	global data_fit 
	global distance
	print "Running analysis..."	
	I = Image.open(str(fname))
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
	fig.savefig(str("./Fitting/"+str(FILENAME)+"X.jpg"))
	
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
	fig.savefig(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
	
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
	fig.savefig(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
	#colorbar()
	#show()
	eqx = "I="+str(Ax)+"exp(-(x-"+str(Bx)+")^2/(2"+str(Cx)+"^2))"
	eqy = "I="+str(Ay)+"exp(-(x-"+str(By)+")^2/(2"+str(Cy)+"^2))"
	file_data.write(str(distance)+" "+str(Ax)+" "+str(Bx)+" "+str(Cx)+" "+str(Ay)+" "+str(By)+" "+str(Cy)+"\n")
	data_fit=[Ax,Bx,Cx,Ay,By,Cy]
	print "Finish"
				
def main():
	a = QApplication(sys.argv)    
	a.setWindowIcon(QtGui.QIcon('./Icon/BP_Icon.png'))
		
	WMP = WMainProfiler()	
	sys.exit(a.exec_())			
   
if __name__=='__main__':
	main()
