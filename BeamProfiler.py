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
matplotlib.rc('legend', fontsize=30)
date = time.strftime("%d%m%y%H%M")
file_data = open(str("./Data/"+date+".dat"),"w")
data_fit=0
FILENAME=date
distance=0
Aback=0
Bback=0
AmpX=0
AmpY=0
WidthX=0
WidthY=0
MeanX=0
MeanY=0

class WidgetContinue(QtGui.QDialog):
	global AmpX,AmpY,WidthX,WidthY,MeanX,MeanY,distance
	def __init__(self, bcapture,bwaist,lineEdit, parent=None):
		self.bcap = bcapture
		self.bwai = bwaist
		self.lEdit = lineEdit
		QtGui.QDialog.__init__(self,parent)
		self.boxV = QtGui.QVBoxLayout()
		self.boxH = QtGui.QHBoxLayout()
		self.boxMain = QtGui.QVBoxLayout()
		self.setWindowTitle("Save...")
		self.lques = QtGui.QLabel()
		self.lques.setText("Do you save this measure?")
		self.byes = QtGui.QPushButton('&Yes')
		self.byes.clicked.connect(self.yes)
		self.bno = QtGui.QPushButton('&No')
		self.bno.clicked.connect(self.no)
		self.boxV.addWidget(self.lques)
		self.boxH.addWidget(self.byes)		
		self.boxH.addWidget(self.bno)				
		self.boxMain.addLayout(self.boxV)
		self.boxMain.addLayout(self.boxH)
		self.setLayout(self.boxMain)
	
	def yes(self):
		self.bcap.setEnabled(True)
		self.bwai.setEnabled(True)
		self.lEdit.setEnabled(True)
		file_data.write(str(distance)+" "+str(AmpX)+" "+str(WidthX)+" "+str(MeanX)+" "+str(AmpY)+" "+str(WidthY)+" "+str(MeanY)+"\n")
		self.hide()
		
	def no(self):
		self.bcap.setEnabled(True)
		self.bwai.setEnabled(True)
		self.lEdit.setEnabled(True)
		self.hide()
		

class WidgetCam(QtGui.QDialog):
	def __init__(self, parent=None):
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
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
		QtGui.QDialog.__init__(self,parent)
		self.setWindowTitle("Control")
		self.labeldistance = QtGui.QLabel()
		self.labeldistance.setText("Distance [mm]")
		self.lineEdit = QtGui.QLineEdit()
		self.lineEdit.resize(100,25)
		self.lineEdit.setFont(QFont("Arial",11))
		self.lineEdit.setValidator(QIntValidator())		
		self.lineEdit.setEnabled(True)
		self.bcapture = QtGui.QPushButton('&Capture', self)
		self.bcapture.clicked.connect(self.capture)
		self.bcapture.setEnabled(True)		
		#self.bback = QtGui.QPushButton('&Backgroud', self)
		#self.bback.clicked.connect(self.back)	
		self.bwaist = QtGui.QPushButton('&Waist', self)
		self.bwaist.clicked.connect(self.fitwaist)
		self.bwaist.setEnabled(True)
		#self.bwaist.setEnabled(False)
		self.breset = QtGui.QPushButton('&Reset', self)
		self.breset.clicked.connect(self.reset)
		self.breset.setEnabled(False)
		layout = QtGui.QVBoxLayout(self)
		layout.addWidget(self.labeldistance)
		layout.addWidget(self.lineEdit)
		layout.addWidget(self.bcapture)		
		#layout.addWidget(self.bback)		
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
		self.wcontinue = WidgetContinue(self.bcapture,self.bwaist,self.lineEdit)
		self.wcontinue.setGeometry(QRect(480, 450, 20, 40))
	
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
		self.wcontinue.show()
		self.bcapture.setEnabled(False)
		self.bwaist.setEnabled(False)
		self.breset.setEnabled(False)
		self.lineEdit.setEnabled(False)
		
	
	#def back(self):
		#global Aback
		#global Bback
		#self.wcam.savePhoto("_background")
		#Ib = Image.open(str("./Fitting/"+str(date)+"_background.jpg"))
		#I1b=Ib.convert('L')
		#ab=np.asarray(I1b,dtype=np.uint8)/255.0
		#Nb=ab.shape[0]
		#Mb=ab.shape[1]
		#max=np.max(ab)
		#xb=np.arange(Mb)
		#yb=np.arange(Nb)
		#Xb,Yb = meshgrid(xb,yb)
		#Ab=[]
		#Bb=[]
		#for i in range(Nb):
		#	for j in range(Mb):
		#		Ab.append([i,j,ab[i][j]])
		#		if ab[i][j] ==max:
		#			Bb.append([i,j,ab[i][j]])
		#Ab = np.asarray(Ab)
		#Bb = np.asarray(Bb)
		#Aback=Ab 
		#Bback=Bb
		#self.bcapture.setEnabled(True)
		#self.bwaist.setEnabled(True)
		#self.bback.setEnabled(False)
		#self.lineEdit.setEnabled(True)
		
	def fitwaist(self):
		global file_data
		global distance
		global FILENAME
		self.bcapture.setEnabled(False)
		self.lineEdit.setEnabled(False)
		self.breset.setEnabled(True)
		file_data.close()
		fig = figure(figsize=(12,10))
		lam=632./(10**6)
		Zd, Ax, Bx, Cx, Ay, By, Cy = loadtxt(str("./Data/"+date+".dat"), unpack = True)
		Bx=Bx
		By=By
		
		def fitw(x, Af, Bf, Cf):##PARA FITEAR CINTURA
			return Af*(np.sqrt(((x-Bf)/Cf)**2+1))
		gmod = Model(fitw)
		#Fitting X Axis
		
		# PARA X
		fig = figure(figsize=(12,10))
		ax2 = fig.gca()
		result = gmod.fit(Bx, x=Zd, Af=0.5, Bf=30 , Cf=350)
		Zresult = result.best_values
		lam=632./(10**6)
		print result.fit_report()
		Af=Zresult.get("Af")
		Wo=Af
		Wo=float("{0:.3f}".format(Wo))
		Bf=Zresult.get("Bf")
		Bf=float("{0:.3f}".format(Bf))
		Cf=Zresult.get("Cf")
		Cf=float("{0:.3f}".format(Cf))
		Zr=np.pi*Wo*Wo/lam
		Zr=float("{0:.3f}".format(Zr))
		#Zr=np.pi*Wo*Wo/lam
		#Wo = np.sqrt((zr*lam)/np.pi)/100 ## /100 Pasando a micras CALCULADO
		#Wo=float("{0:.6f}".format(Wo))
		Div = lam/(np.pi*Wo)
		Div=float("{0:.5f}".format(Div))
		errWo=0.018
		errZr=(2*np.pi*Wo/lam)*errWo
		errZr=float("{0:.2f}".format(errZr))
		errDiv=(lam*errWo)/(np.pi*Wo*Wo)
		errDiv=float("{0:.5f}".format(errDiv))
		par=result.params["Af"]
		plt.xlabel(r"$Z (mm)$", fontsize = 27, color = (0,0,0))
		plt.ylabel(r"$W_x (\mu m)$", fontsize = 27, color = (0,0,0))
		Eqx=str(Af)+"\sqrt${1+\frac{(x-"+str(Bf)+")^2}{"+str(Cf)+"}}"
		#plt.text(x = 1, y = 0.0, s = r''+Eqx+'' , fontsize = 24)
		plt.text(250, -0.08, r"$W_x$: "+str(Wo)+" $\pm$ "+str(errWo)+" $\mu m$\n$Z_r$: "+str(Zr)+" $\pm$ "+str(errZr)+" $mm$ \n$\Theta$: "+str(Div)+" $\pm$ "+str(errDiv)+" $rad$", style='italic', fontsize = 24,
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
		plt.grid()
		X = np.linspace(-800, 800, 1000, endpoint=True)
		plt.title(r'$Ancho\, horizontal\, del\, haz\, en\, funci\acute{o}n\, de\, la\, posici\acute{o}n.$', fontsize = 45)
		plt.plot(X,fitw(X, Af, Bf, Cf),linestyle="--",linewidth = 3,color = (0,1,0.5))
		plt.plot(X,fitw(X, -Af, Bf, Cf),linestyle="--",linewidth = 3,color = (0,1,0.5))
		plt.plot(Zd,Bx,'.',markersize=15,color = (1,0,0))
		plt.errorbar(Zd,Bx,xerr=1,linestyle='None',fmt="--o")
		fig.set_size_inches(18.5, 10.5)		
		fig.savefig(str("./Fitting/"+str(date)+"WaistX.png"))
		self.waistx.setPicture(str("./Fitting/"+str(date)+"WaistX.png"))
		self.waistx.setGeometry(QRect(10, 60, 350, 200))
		self.waistx.show()
		
		
		(Af, Bf, Cf), _ = curve_fit(fitw, Zd, By)
		fig2 = figure(figsize=(12,10))
		result = gmod.fit(By, x=Zd, Af=0.5, Bf=30 , Cf=350)
		Zresult = result.best_values
		lam=632./(10**6)
		Af=Zresult.get("Af")
		Wo=Af
		Wo=float("{0:.3f}".format(Wo))
		Bf=Zresult.get("Bf")
		Bf=float("{0:.3f}".format(Bf))
		Cf=Zresult.get("Cf")
		Cf=float("{0:.3f}".format(Cf))
		Zr=np.pi*Wo*Wo/lam
		Zr=float("{0:.3f}".format(Zr))
		#Zr=np.pi*Wo*Wo/lam
		#Wo = np.sqrt((zr*lam)/np.pi)/100 ## /100 Pasando a micras CALCULADO
		#Wo=float("{0:.6f}".format(Wo))
		print result.fit_report()
		Div = lam/(np.pi*Wo)
		Div=float("{0:.5f}".format(Div))
		errWo=0.022
		errZr=(2*np.pi*Wo/lam)*errWo
		errZr=float("{0:.2f}".format(errZr))
		errDiv=(lam*errWo)/(np.pi*Wo*Wo)
		errDiv=float("{0:.5f}".format(errDiv))
		par=result.params["Af"]
		plt.xlabel(r"$Z (mm)$", fontsize = 27, color = (0,0,0))
		plt.ylabel(r"$W_y (\mu m)$", fontsize = 27, color = (0,0,0))
		Eqx=str(Af)+"\sqrt${1+\frac{(y-"+str(Bf)+")^2}{"+str(Cf)+"}}"
		#plt.text(x = 1, y = 0.0, s = r''+Eqx+'' , fontsize = 24)
		plt.text(350, -0.08, r"$W_y$: "+str(Wo)+" $\pm$ "+str(errWo)+" $\mu m$\n$Z_r$: "+str(Zr)+" $\pm$ "+str(errZr)+" $mm$ \n$\Theta$: "+str(Div)+" $\pm$ "+str(errDiv)+" $rad$", style='italic', fontsize = 24,
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
		plt.grid()
		plt.title(r'$Ancho\, vertical\, del\, haz\, en\, funci\acute{o}n\, de\, la\, posici\acute{o}n.$', fontsize = 45)
		X = np.linspace(-800, 800, 1000, endpoint=True)
		plt.plot(X,fitw(X, Af, Bf, Cf),linestyle="--",linewidth = 3,color = (0,1,0.5))
		plt.plot(X,fitw(X, -Af, Bf, Cf),linestyle="--",linewidth = 3,color = (0,1,0.5))
		plt.plot(Zd,By,'.',markersize=15,color = (1,0,0))
		plt.errorbar(Zd,By,xerr=1,linestyle='None',fmt="--o")
		fig2.set_size_inches(18.5, 10.5)		
		fig2.savefig(str("./Fitting/"+str(date)+"WaistY.png"))
		self.waisty.setPicture(str("./Fitting/"+str(date)+"WaistY.png"))
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
		self.table.setRowCount(7)
		self.table.setColumnCount(2)
		self.table.setItem(0,0, QTableWidgetItem('Distance [mm]'))		
		self.table.setItem(1,0, QTableWidgetItem("Amp x [mm]"))
		self.table.setItem(2,0, QTableWidgetItem("Width x [mm]"))
		self.table.setItem(3,0, QTableWidgetItem("Mean x [mm]"))
		self.table.setItem(4,0, QTableWidgetItem("Amp y [mm]"))
		self.table.setItem(5,0, QTableWidgetItem("Width y [mm]"))
		self.table.setItem(6,0, QTableWidgetItem("Mean y [mm]"))
		self.setLayout(self.layout)	
	
	def setValues(self):
		self.table.setItem(0,1, QTableWidgetItem(str(data_fit[0])))
		self.table.setItem(1,1, QTableWidgetItem(str(data_fit[1])))
		self.table.setItem(2,1, QTableWidgetItem(str(data_fit[2])))
		self.table.setItem(3,1, QTableWidgetItem(str(data_fit[3])))
		self.table.setItem(4,1, QTableWidgetItem(str(data_fit[4])))
		self.table.setItem(5,1, QTableWidgetItem(str(data_fit[5])))
		self.table.setItem(6,1, QTableWidgetItem(str(data_fit[6])))
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
		
		self.wdata.setGeometry(QRect(880, 120, 239, 255))
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
	global AmpX,AmpY,WidthX,WidthY,MeanX,MeanY
	global distance
	global data_fit 
	global date
	print "Running analysis..."	
	#R = Image.open("ruido.jpg")
	#R = Image.open("./Fitting/"+str(date)+"_background.jpg")
	#R1=R.convert('L') #Convert to Gray Scale
	I = Image.open(fname)
	I1=I.convert('L') #Convert to Gray Scale
	#I1.show()
	#Ra=np.asarray(R1,dtype=np.uint8)/255.0
	a=np.asarray(I1,dtype=np.uint8)/255.0 #convierte I1 en una matriz
	#a=a-Ra
	N=a.shape[0]
	M=a.shape[1]
	a=a[2:N-2,2:M-2]
	N=N-4
	M=M-4	
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
	def gaussian(x, amp, cen, wid, hei):
		#"1-d gaussian: gaussian(x, amp, cen, wid)"
		return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2)) + hei

	gmod = Model(gaussian)
	fa=0.0093  #Factor de Ajuste px to mm
	#Fitting X Axis
	
	result = gmod.fit(D[:,1], x=D[:,0]*fa-0.13, amp=1, cen=xm*fa-0.13, wid=50*fa, hei=0.1)
	Xresult = result.best_values
	print "Amp: "+str(Xresult.get("amp"))+", Width: "+str(Xresult.get("wid"))+", Mean: "+str(Xresult.get("cen"))+", Height: "+str(Xresult.get("hei"))
	print result.fit_report()
	fig = figure(figsize=(12,10))
	ax1 = fig.gca()
	plt.grid()
	plt.plot(D[:,0]*fa-0.13,D[:,1],'.',markersize=10,color = (0,0,1),label="Data")
	plt.plot(D[:,0]*fa-0.13, result.best_fit, 'r-',linestyle="--",linewidth = 5,color = (0,1,0.5),label="Gaussian Fit")	
	plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
	#plt.plot(D[:,0],func(D[:,0], Af, Bf, Cf),linewidth = 5)
	ax1.set_xlabel(r'$x\, [\mu m]$')
	ax1.set_ylabel(r'$Intensity$')
	Ampx=Xresult.get("amp")
	Widthx=Xresult.get("wid")
	Meanx=Xresult.get("cen")
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"X.jpg"))
	
	#Fitting Y Axis
	result = gmod.fit(C[:,1], x=C[:,0]*fa, amp=1, cen=ym*fa, wid=50*fa, hei=0.1)
	Yresult = result.best_values
	print "Amp: "+str(Yresult.get("amp"))+", Width: "+str(Yresult.get("wid"))+", Mean: "+str(Yresult.get("cen"))+", Height: "+str(Yresult.get("hei"))
	print result.fit_report()
	fig = figure(figsize=(12,10))
	ax2 = fig.gca()
	plt.grid()
	plt.plot(C[:,0]*fa,C[:,1],'.',markersize=10,color = (0,0,1),label="Data")
	plt.plot(C[:,0]*fa, result.best_fit, 'r-',linestyle="--",linewidth = 5,color = (0,1,0.5),label="Gaussian Fit")	
	plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
	#plt.plot(D[:,0],func(D[:,0], Af, Bf, Cf),linewidth = 5)
	ax2.set_xlabel(r'$y\, [\mu m]$')
	ax2.set_ylabel(r'$Intensity$')
	Ampy=Yresult.get("amp")
	Widthy=Yresult.get("wid")
	Meany=Yresult.get("cen")
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"Y.jpg"))
	
	#fig=figure(figsize = (20,20))	
	#ax = fig.gca(projection='3d')
	#figura = ax.plot_surface(X,Y,a,cmap = cm.jet,linewidth = 0)
	#fig.colorbar(figura, shrink=0.5, aspect=5)  
	#ax.set_xlabel("x")
	#ax.set_ylabel("y")
	#ax.set_zlabel("Intensity")
	#ax.view_init(elev=45, azim=45)
	#cset = ax.contour(X, Y, a, zdir='x', offset=-20, cmap=cm.jet)
	#cset = ax.contour(X, Y, a, zdir='y', offset=-20, cmap=cm.jet)
	#fig.set_size_inches(18.5, 10.5)
	#fig.savefig(str("./Fitting/"+str(FILENAME)+"3D.jpg"))
	#colorbar()
	
	fig=figure(figsize = (20,20))	
	ax = fig.gca()#projection='3d')
	ax.set_xlabel("x [px]")
	ax.set_ylabel("y [px]")
	a2d=a
	a2d[int(xm):int(xm+1)] = int(1)
	a2d[:,int(ym):int(ym+1)] = int(1)
	plt.imshow(a2d)
	fig.set_size_inches(18.5, 10.5)
	fig.savefig(str("./Fitting/"+str(FILENAME)+"3D.jpg"))	
	
	eqx = "I="+str(Ampx)+"exp(-(x-"+str(Widthx)+")^2/(2"+str(Meanx)+"^2))"
	eqy = "I="+str(Ampy)+"exp(-(x-"+str(Widthy)+")^2/(2"+str(Meany)+"^2))"	
	Widthx=Widthx*2
	Widthy=Widthy*2	
	data_fit=[distance,Ampx,Widthx,Meanx,Ampy,Widthy,Meany]
	#print(result.best_values)
	AmpX,AmpY,WidthX,WidthY,MeanX,MeanY=Ampx,Ampy,Widthx,Widthy,Meanx,Meany
	print "Finish"
				
def main():
	app = QApplication(sys.argv)    
	app.setWindowIcon(QtGui.QIcon('./Icon/BP_Icon.svg'))
		
	WMP = WMainProfiler()	
	sys.exit(app.exec_())			
   
if __name__=='__main__':
	main()
