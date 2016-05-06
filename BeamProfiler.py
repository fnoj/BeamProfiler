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
from lmfit import Model

matplotlib.rc('xtick', labelsize=45) 
matplotlib.rc('ytick', labelsize=45)
matplotlib.rc('axes', labelsize=45)
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
date = time.strftime("%d%m%y%H%M")
file_data = open(str("./Data/"+date+".dat"),"w")
data_fit=0
distance=0
FILENAME=date
Aback=0
Bback=0

class WidgetCam(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self,parent)
		self.setWindowTitle("Profile")
		self.hBoxP = QHBoxLayout()
		self.scene = QtGui.QGraphicsScene(self)
		#self.scene.addText("Hello, world!");
		self.profile = QtGui.QGraphicsView(self.scene)
		self.profile.setFixedWidth(644/2)
		self.profile.setFixedHeight(484/2)
		self.hBoxP.addWidget(self.profile)
		self.setLayout(self.hBoxP)
		
	def ProcessFrame(self, im):
		#print "Frame update", im
		self.pix = QtGui.QPixmap(im)
		self.scene.clear()
		self.scene.addPixmap(self.pix)
	
	def savePhoto(self,name):
		self.pixMap = QPixmap.grabWidget(self.profile);
		self.pixMap.save(str("./Fitting/"+str(FILENAME)+str(name)+".jpg"))
		
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
		self.labeldistance = QtGui.QLabel()
		self.labeldistance.setText("Distance [mm]")
		self.lineEdit = QtGui.QLineEdit()
		self.lineEdit.resize(100,25)
		self.lineEdit.setFont(QFont("Arial",11))
		self.lineEdit.setValidator(QIntValidator())		
		self.lineEdit.setEnabled(False)
		self.bcapture = QtGui.QPushButton('&Capture', self)
		self.bcapture.clicked.connect(self.capture)
		self.bcapture.setEnabled(False)
		self.bback = QtGui.QPushButton('&Backgroud', self)
		self.bback.clicked.connect(self.back)	
		self.bwaist = QtGui.QPushButton('&Waist', self)
		self.bwaist.clicked.connect(self.fitwaist)
		self.bwaist.setEnabled(False)
		self.breset = QtGui.QPushButton('&Reset', self)
		self.breset.clicked.connect(self.reset)
		self.breset.setEnabled(False)
		layout = QtGui.QVBoxLayout(self)
		layout.addWidget(self.labeldistance)
		layout.addWidget(self.lineEdit)
		layout.addWidget(self.bcapture)		
		layout.addWidget(self.bback)		
		layout.addWidget(self.bwaist)
		layout.addWidget(self.breset)
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
		print "Profile saved as ./Fitting/"+FILENAME+"P.jpg"
		self.wcam.savePhoto("P")
		analice(str("./Fitting/"+str(FILENAME)+"P.jpg"))
		self.wfity.setPicture(str("./Fitting/"+str(FILENAME)+"X.jpg"))
		self.wfitx.setPicture(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
		self.wfit3d.setPicture(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
		self.wdata.setValues()
	
	def back(self):
		global Aback
		global Bback
		self.wcam.savePhoto("_background")
		Ib = Image.open(str("./Fitting/"+str(date)+"_background.jpg"))
		I1b=Ib.convert('L')
		ab=np.asarray(I1b,dtype=np.uint8)/255.0
		Nb=ab.shape[0]
		Mb=ab.shape[1]
		max=np.max(ab)
		xb=np.arange(Mb)
		yb=np.arange(Nb)
		Xb,Yb = meshgrid(xb,yb)
		Ab=[]
		Bb=[]
		for i in range(Nb):
			for j in range(Mb):
				Ab.append([i,j,ab[i][j]])
				if ab[i][j] ==max:
					Bb.append([i,j,ab[i][j]])
		Ab = np.asarray(Ab)
		Bb = np.asarray(Bb)
		Aback=Ab 
		Bback=Bb
		self.bcapture.setEnabled(True)
		self.bwaist.setEnabled(True)
		self.bback.setEnabled(False)
		self.lineEdit.setEnabled(True)
		
	def fitwaist(self):
		global file_data
		global FILENAME
		self.bcapture.setEnabled(False)
		self.lineEdit.setEnabled(False)
		self.breset.setEnabled(True)
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
		self.waistx.setGeometry(QRect(10, 60, 350, 200))
		self.waistx.show()
		
		fig2 = figure(figsize=(12,10))
		plt.plot(Zd,Ay,'.')
		plt.plot(Zd,func(Zd, Af, Bf, Cf, Df),linewidth = 2)
		zr=Cf
		Wo = np.sqrt((zr*lam)/np.pi)
		Y = np.linspace(-600, 600, 1000, endpoint=True)
		plt.plot(Y,func(Y, Af, Bf, Cf, Df),linewidth = 1)
		plt.plot(Y,func(Y, -Af, Bf, Cf, Df),linewidth = 1)
		fig2.savefig(str("./Fitting/"+str(FILENAME)+"WaistY.jpg"))
		self.waisty.setPicture(str("./Fitting/"+str(FILENAME)+"WaistY.jpg"))
		self.waisty.setGeometry(QRect(60, 120, 350, 200))
		self.waisty.show()
				
	def reset(self):
		file_data = open(str("./Data/"+date+".dat"),"w")
		self.bback.setEnabled(True)
		self.breset.setEnabled(False)
		self.bwaist.setEnabled(False)
		
class WidgetData(QtGui.QDialog):
	def __init__(self, parent=None):
		super(WidgetData, self).__init__(parent)
		self.setWindowTitle("Data")
		self.layout = QtGui.QGridLayout() 
		self.table = QtGui.QTableWidget()
		self.layout.addWidget(self.table)
		self.table.setRowCount(6)
		self.table.setColumnCount(2)
		self.table.setItem(0,0, QTableWidgetItem("Amp x"))
		self.table.setItem(1,0, QTableWidgetItem("Width x"))
		self.table.setItem(2,0, QTableWidgetItem("Mean x"))
		self.table.setItem(3,0, QTableWidgetItem("Amp y"))
		self.table.setItem(4,0, QTableWidgetItem("Width y"))
		self.table.setItem(5,0, QTableWidgetItem("Mean y"))
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
	global date
	print "Running analysis..."	
	fname="lena.jpg"
	R = Image.open("ruido.jpg")
	#R = Image.open("./Fitting/"+str(date)+"_background.jpg")
	R1=R.convert('L') #Convert to Gray Scale
	I = Image.open(fname)
	I1=I.convert('L') #Convert to Gray Scale
	#I1.show()
	Ra=np.asarray(R1,dtype=np.uint8)/255.0
	a=np.asarray(I1,dtype=np.uint8)/255.0 #convierte I1 en una matriz

	a=a-Ra
	N=a.shape[0]
	M=a.shape[1]
	max=np.max(a)
	x=np.arange(N)
	y=np.arange(M)
	X,Y = meshgrid(y,x)
	A=[]
	B=[]
	for i in range(N):
		for j in range(M):
			A.append([i,j,a[i][j]])
			if a[i][j] ==max:
				B.append([i,j,a[i][j]])
        
	B = np.asarray(B)
	A = np.asarray(A)
	#Sacar puntos max y min en x y y - Se toman los maximos a lo largo del eje vertical y Horizontal
	xm=int(abs(np.max (B[:,0])-np.min(B[:,0]))/2)+np.min(B[:,0])
	ym=int(abs(np.max (B[:,1])-np.min(B[:,1]))/2)+np.min(B[:,1])
	
	C=[]
	D=[]
	for i in range(N*M):
	    if A[i,0]==xm:
	        C.append([A[i,1],A[i,2]])
	    if A[i,1]==ym:
	        D.append([A[i,0],A[i,2]])            
	
	C=np.asarray(C)  # Para x
	D=np.asarray(D)  # Para y
	def gaussian(x, amp, cen, wid):
		#"1-d gaussian: gaussian(x, amp, cen, wid)"
		return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

	gmod = Model(gaussian)
	#Fitting X Axis
	result = gmod.fit(D[:,1], x=D[:,0], amp=100, cen=150, wid=50)
	Xresult = result.best_values
	print "Amp: "+str(Xresult.get("amp"))+", Width: "+str(Xresult.get("wid"))+", Mean: "+str(Xresult.get("cen"))

	fig = figure(figsize=(12,10))
	ax1 = fig.gca()
	plt.plot(D[:,0],D[:,1],'.')
	plt.plot(D[:,0], result.best_fit, 'r-',linewidth = 5)
	#plt.plot(D[:,0],func(D[:,0], Af, Bf, Cf),linewidth = 5)
	ax1.set_xlabel("x (px)")
	ax1.set_ylabel("Intensity")
	Ampx=Xresult.get("amp")
	Widthx=Xresult.get("wid")
	Meanx=Xresult.get("cen")
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"X.jpg"))
	
	
	#Fitting Y Axis
	result = gmod.fit(C[:,1], x=C[:,0], amp=100, cen=150, wid=50)
	Yresult = result.best_values
	print "Amp: "+str(Yresult.get("amp"))+", Width: "+str(Yresult.get("wid"))+", Mean: "+str(Yresult.get("cen"))
	fig = figure(figsize=(12,10))
	ax2 = fig.gca()
	plt.plot(C[:,0],C[:,1],'.')
	plt.plot(C[:,0], result.best_fit, 'r-',linewidth = 5)
	ax2.set_xlabel("y (px)")
	ax2.set_ylabel("Intensity")
	Ampy=Yresult.get("amp")
	Widthy=Yresult.get("wid")
	Meany=Yresult.get("cen")
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
	
	fig=figure(figsize = (20,20))
	ax = fig.gca(projection='3d')
	figura = ax.plot_surface(X,Y,a,cmap = cm.jet,linewidth = 0)
	fig.colorbar(figura, shrink=0.5, aspect=5)  
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("Intensity")
	ax.view_init(elev=45, azim=45)
	cset = ax.contour(X, Y, a, zdir='x', offset=-20, cmap=cm.jet)
	cset = ax.contour(X, Y, a, zdir='y', offset=-20, cmap=cm.jet)
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
	#colorbar()
	eqx = "I="+str(Ampx)+"exp(-(x-"+str(Widthx)+")^2/(2"+str(Meanx)+"^2))"
	eqy = "I="+str(Ampy)+"exp(-(x-"+str(Widthy)+")^2/(2"+str(Meany)+"^2))"
	file_data.write(str(distance)+" "+str(Ampx)+" "+str(Widthx)+" "+str(Meanx)+" "+str(Ampy)+" "+str(Widthy)+" "+str(Meany)+"\n")
	data_fit=[Ampx,Widthx,Meanx,Ampy,Widthy,Meany]
	#print(result.best_values)
	print "Finish"
				
def main():
	a = QApplication(sys.argv)    
	a.setWindowIcon(QtGui.QIcon('./Icon/BP_Icon.png'))
		
	WMP = WMainProfiler()	
	sys.exit(a.exec_())			
   
if __name__=='__main__':
	main()

