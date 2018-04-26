/* ******************************************************************************
* Description: Panel with application parameter fields.
*
*  Version 1.0.0
*  Date: Jul 17, 2008
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _PARAMETERTAB_
#define _PARAMETERTAB_

#include <QObject>
#include <QDialog>
#include <QtGui/QWidget>
#include <QtGui/QTabWidget>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpinBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>

#include "qt/imagepanel.h"
#include "lensBlurApp.hpp"
#include "imageBuffer.hpp"

//===============================================================
// Description: Modal dialog to select kernel.
//===============================================================
class KernelDialog : public QDialog
//===============================================================
{
	Q_OBJECT
private:
	QLabel**		kernelIcons;
	QRadioButton**	kernels;
	QPushButton*	okButton;
public:

//-----------------------------------------------------------------
// Summary: Constructs the kernel selection dialog.
// Arguments: selectedKernel - index of the current kernel
//-----------------------------------------------------------------
KernelDialog(int selectedKernel) {
//-----------------------------------------------------------------
	kernelIcons = new QLabel*[8];
	int k=0;
	for(;k<7;k++) {
		char iconFilePath[128];
		sprintf_s(iconFilePath,128,"resources/kernel%d_icon.bmp",(k+1));
		QImage icon(iconFilePath);

		kernelIcons[k] = new QLabel();
		if (!icon.isNull()) {
			kernelIcons[k]->setPixmap(QPixmap::fromImage(icon));
		} else {
			printf("[WARN] Could not load kernel icon: %s\n",iconFilePath);
		}
	}
	kernelIcons[k] = 0;

	kernels = new QRadioButton*[8];
	kernels[0]  = new QRadioButton("triangle");
	kernels[1]  = new QRadioButton("square");
	kernels[2]  = new QRadioButton("pentagon");
	kernels[3]  = new QRadioButton("hexagon");
	kernels[4]  = new QRadioButton("septagon");
	kernels[5]  = new QRadioButton("octagon");
	kernels[6]  = new QRadioButton("cycle");
	kernels[7]  = 0;

	okButton = new QPushButton("OK");
	connect(okButton, SIGNAL(clicked()), SLOT(close()));

	QGridLayout *mainLayout = new QGridLayout();
	int i=0;
	for(;kernels[i];i++) {
		mainLayout->addWidget(kernelIcons[i],2*(i/3),i%3);
		mainLayout->addWidget(kernels[i],2*(i/3)+1,i%3);
		if (i+1 == selectedKernel) {
			kernels[i]->setChecked(true);
		}
	}
	mainLayout->addWidget(okButton,2*((i+2)/3)+1,1);
	setLayout(mainLayout);

	setWindowTitle(tr("Select mask"));
}

//-----------------------------------------------------------------
// Returns: index of the selected kernel.
//-----------------------------------------------------------------
int getKernel() {
//-----------------------------------------------------------------
	for(int i=0;kernels[i];i++) {
		if (kernels[i]->isChecked()) {
			return i;
		}
	}
	return 0;
}

};

//===============================================================
// Description: handler of the 'Basic' tab.
//===============================================================
class BasicParametersPanel : public QWidget
//===============================================================
{
	Q_OBJECT
private:
	LensBlurApp*	lensBlurApp;
	Parameters*		parameters;
	ImagePanel*		imagePanel;

	QLabel*			modeLabel;
	QComboBox*		mode;

	QWidget**		artistWidgets;
	QWidget**		physicalWidgets;

	// artist
	QLabel*			blurStrengthLabel;
	QSpinBox*		blurStrength;
	QLabel*			epsilonLabel;
	QDoubleSpinBox*	epsilon;
	QLabel*			overlapLabel;
	QDoubleSpinBox*	overlap;
	// physic
	QLabel*			fLengthLabel;
	QDoubleSpinBox*	fLength;
	QLabel*			msLabel;
	QDoubleSpinBox*	ms;
	QLabel*			fStopLabel;
	QComboBox*		fStop;
	QLabel*			imageRangeLabel;
	QDoubleSpinBox*	imageRange;
	QLabel*			cameraDistanceLabel;
	QDoubleSpinBox*	cameraDistance;
	// common
	QLabel*			kernelLabel;
	QPushButton*	kernelButton;
	QLabel*			focusLabel;
	QSpinBox*		focus;
	QLabel*			bloomAmountLabel;
	QSpinBox*		bloomAmount;
	QLabel*			bloomThresholdLabel;
	QSpinBox*		bloomThreshold;

	QPushButton*	renderButton;

public:

//-----------------------------------------------------------------
// Summary: Constructs the kernel parameter handler.
// Arguments: lensBlurApp - application controller
//            imagePanel - image panel
//-----------------------------------------------------------------
BasicParametersPanel(LensBlurApp* lensBlurApp,ImagePanel* imagePanel) {
//-----------------------------------------------------------------
	this->lensBlurApp = lensBlurApp;
	this->parameters = &(lensBlurApp->parameters);
	this->imagePanel = imagePanel;

	modeLabel = new QLabel("Mode:");
	mode = new QComboBox();
	mode->addItem(QString("Artist"));
	mode->addItem(QString("Physical"));
	mode->setCurrentIndex(parameters->getMode());
	connect(mode, SIGNAL(currentIndexChanged(int)), SLOT(refreshByMode()));

	blurStrengthLabel = new QLabel("Blur strength (0 - 50):");
	blurStrength = new QSpinBox();
	blurStrength->setRange(0,50);
	blurStrength->setValue(parameters->getStrengthAsControl());

	fLengthLabel = new QLabel("fLength:");
	fLength = new QDoubleSpinBox();
	fLength->setRange(18.0f,600.0f);
	fLength->setValue(parameters->getFLength());

	msLabel = new QLabel("ms:");
	ms = new QDoubleSpinBox();
	ms->setRange(0.5f,4.0f);
	ms->setValue(parameters->getMS());

	fStopLabel = new QLabel("fStop:");
	fStop = new QComboBox();
	fStop->addItem(QString("f/1"));
	fStop->addItem(QString("f/1.4"));
	fStop->addItem(QString("f/2"));
	fStop->addItem(QString("f/2.8"));
	fStop->addItem(QString("f/4"));
	fStop->addItem(QString("f/5.6"));
	fStop->addItem(QString("f/8"));
	fStop->addItem(QString("f/11"));
	fStop->addItem(QString("f/16"));
	fStop->addItem(QString("f/22"));
	fStop->addItem(QString("f/32"));
	fStop->addItem(QString("f/45"));
	fStop->setCurrentIndex(parameters->getFStop());

	imageRangeLabel = new QLabel("Image range (m):");
	imageRange = new QDoubleSpinBox();
	imageRange->setRange(0.0f,200.0f);
	imageRange->setValue(parameters->getDistance());

	cameraDistanceLabel = new QLabel("Camera distance (m):");
	cameraDistance = new QDoubleSpinBox();
	cameraDistance->setRange(0.0f,200.0f);
	cameraDistance->setValue(parameters->getCameraDistance());

	artistWidgets = new QWidget*[3];
	artistWidgets[0] = blurStrengthLabel;
	artistWidgets[1] = blurStrength;
	artistWidgets[2] = 0;
	physicalWidgets = new QWidget*[11];
	physicalWidgets[0] = fLengthLabel;
	physicalWidgets[1] = fLength;
	physicalWidgets[2] = msLabel;
	physicalWidgets[3] = ms;
	physicalWidgets[4] = fStopLabel;
	physicalWidgets[5] = fStop;
	physicalWidgets[6] = imageRangeLabel;
	physicalWidgets[7] = imageRange;
	physicalWidgets[8] = cameraDistanceLabel;
	physicalWidgets[9] = cameraDistance;
	physicalWidgets[10] = 0;

	QGridLayout *mainLayout = new QGridLayout();
	mainLayout->addWidget(modeLabel,0,0);
	mainLayout->addWidget(mode,0,1);
	// artist
	int i;
	for(i=0;artistWidgets[i];i++) {
		mainLayout->addWidget(artistWidgets[i],i/2+1,i%2);
	}
	// physic
	for(i=0;physicalWidgets[i];i++) {
		mainLayout->addWidget(physicalWidgets[i],i/2+4,i%2);
	}
	// common
	epsilonLabel = new QLabel("Z epsilon:");
	epsilon = new QDoubleSpinBox();
	epsilon->setRange(0.0f,1.0f);
	epsilon->setValue(parameters->getEpsilonScale());

	overlapLabel = new QLabel("Smoothing:");
	overlap = new QDoubleSpinBox();
	overlap->setRange(0.0f,1.0f);
	overlap->setValue(parameters->getOverlap());

	kernelLabel = new QLabel("Mask:");
	kernelButton = new QPushButton();
	setKernelButtonIcon(lensBlurApp->apKernel->type);
	connect(kernelButton, SIGNAL(clicked()), SLOT(selectKernel()));

	focusLabel = new QLabel("Focal point:");
	focus = new QSpinBox();
	focus->setRange(0,255);
	focus->setValue(parameters->getFocus());

	bloomAmountLabel = new QLabel("Bloom amount:");
	bloomAmount = new QSpinBox();
	bloomAmount->setRange(0,100);
	bloomAmount->setValue(parameters->getBloomAmount());

	bloomThresholdLabel = new QLabel("Bloom threshold:");
	bloomThreshold = new QSpinBox();
	bloomThreshold->setRange(0,255);
	bloomThreshold->setValue(parameters->getThreshold());

	renderButton = new QPushButton("Render");
	connect(renderButton, SIGNAL(clicked()), SLOT(rerender()));

	mainLayout->addWidget(epsilonLabel,i/2+4,0);
	mainLayout->addWidget(epsilon,i/2+4,1);
	mainLayout->addWidget(overlapLabel,i/2+5,0);
	mainLayout->addWidget(overlap,i/2+5,1);
	mainLayout->addWidget(kernelLabel,i/2+6,0);
	mainLayout->addWidget(kernelButton,i/2+6,1);
	mainLayout->addWidget(focusLabel,i/2+7,0);
	mainLayout->addWidget(focus,i/2+7,1);
	mainLayout->addWidget(bloomAmountLabel,i/2+8,0);
	mainLayout->addWidget(bloomAmount,i/2+8,1);
	mainLayout->addWidget(bloomThresholdLabel,i/2+9,0);
	mainLayout->addWidget(bloomThreshold,i/2+9,1);

	mainLayout->addWidget(renderButton,i/2+10,1);

	setLayout(mainLayout);

	refreshByMode();
}

//-----------------------------------------------------------------
// Summary: displays the kernel icon image.
// Arguments: type - kernel index
//-----------------------------------------------------------------
void setKernelButtonIcon(int type) {
//-----------------------------------------------------------------
	char iconFilePath[128];
	sprintf_s(iconFilePath,128,"resources/kernel%d_icon.bmp",type);
	QIcon kernelIcon(iconFilePath);
	if (!kernelIcon.isNull()) {
		kernelButton->setFixedSize(64,64);
		kernelButton->setIconSize(QSize(56,56));
		kernelButton->setIcon(kernelIcon);
		kernelButton->setText("");
	} else {
		kernelButton->setIcon(kernelIcon);
		char typeText[32];
		switch(type) {
			case 1: sprintf_s(typeText,32,"triangle"); break;
			case 2: sprintf_s(typeText,32,"squar"); break;
			case 3: sprintf_s(typeText,32,"pentagon"); break;
			case 4: sprintf_s(typeText,32,"hexagon"); break;
			case 5: sprintf_s(typeText,32,"septagon"); break;
			case 6: sprintf_s(typeText,32,"ocatagon"); break;
			case 7: sprintf_s(typeText,32,"cycle"); break;
			default: sprintf_s(typeText,32,"custom"); break;
		}
		kernelButton->setText(typeText);
	}
}

//-----------------------------------------------------------------
// Summary: reads the settings from the GUI.
//-----------------------------------------------------------------
void readParameters() {
//-----------------------------------------------------------------
	parameters->setMode(mode->currentIndex());
	// artist
	parameters->setStrength(100.0f*(float)blurStrength->value()/(float)blurStrength->maximum());
	parameters->setEpsilon((float)epsilon->value());
	parameters->setOverlap((float)overlap->value());
	// physic
	parameters->setFLength((float)fLength->value());
	parameters->setMS((float)ms->value());
	parameters->setFStop((float)fStop->currentIndex());
	parameters->setDistance((float)imageRange->value());
	parameters->setCameraDistance((float)cameraDistance->value());
	// common
	parameters->setFocus(focus->value());
	parameters->setBloomAmount(bloomAmount->value());
	parameters->setThreshold(bloomThreshold->value());
}

//-----------------------------------------------------------------
// Returns: the focus widget.
//-----------------------------------------------------------------
QSpinBox* getFocusWidget() {
//-----------------------------------------------------------------
	return focus;
}

private slots:

//-----------------------------------------------------------------
// Summary: draws the widgets depends on the selected calculation mode.
//-----------------------------------------------------------------
void refreshByMode() {
//-----------------------------------------------------------------
	QWidget** visibleWidgets = 0;
	QWidget** invisibleWidgets = 0;
	switch(mode->currentIndex()) {
		case 0: visibleWidgets = artistWidgets; invisibleWidgets = physicalWidgets; break;
		case 1: visibleWidgets = physicalWidgets; invisibleWidgets = artistWidgets; break;
	}
	if (visibleWidgets) {
		for(int i=0;visibleWidgets[i];i++) {
			visibleWidgets[i]->setVisible(true);
		}
	}
	if (invisibleWidgets) {
		for(int i=0;invisibleWidgets[i];i++) {
			invisibleWidgets[i]->setVisible(false);
		}
	}
}

//-----------------------------------------------------------------
// Summary: opens the kernel selection dialog.
//-----------------------------------------------------------------
void selectKernel() {
//-----------------------------------------------------------------
	KernelDialog kernelDialog(lensBlurApp->apKernel->type);
	kernelDialog.exec();
	int kernel = kernelDialog.getKernel()+1;
	// load kernel
	lensBlurApp->changeKernel(kernel);
	// refresh kernel icon
	setKernelButtonIcon(kernel);
}

//-----------------------------------------------------------------
// Summary: renders the image by the settings.
//-----------------------------------------------------------------
void rerender() {
//-----------------------------------------------------------------
	readParameters();
	imagePanel->rerenderImage();
}

};

//===============================================================
// Description: handler of the 'Animation' tab.
//===============================================================
class AnimParametersPanel : public QWidget
//===============================================================
{
	Q_OBJECT
private: 
	LensBlurApp*	lensBlurApp;
	ImagePanel*		imagePanel;

	QLabel*			framesLabel;
	QSpinBox*		frames;
	QLabel*			focusStartLabel;
	QSpinBox*		focusStart;
	QLabel*			focusEndLabel;
	QSpinBox*		focusEnd;

	QPushButton*	animButton;

public:

//-----------------------------------------------------------------
// Summary: Constructs the kernel parameter handler.
// Arguments: lensBlurApp - application controller
//            imagePanel - image panel
//-----------------------------------------------------------------
AnimParametersPanel(LensBlurApp* lensBlurApp, ImagePanel* imagePanel) {
//-----------------------------------------------------------------
	this->lensBlurApp = lensBlurApp;
	this->imagePanel = imagePanel;

	framesLabel = new QLabel("Number of frames:");
	frames = new QSpinBox();
	frames->setRange(1,100);
	frames->setValue(10);

	focusStartLabel = new QLabel("Focus at first frame:");
	focusStart = new QSpinBox();
	focusStart->setRange(0,255);
	focusStart->setValue(0);

	focusEndLabel = new QLabel("Focus at last frame:");
	focusEnd = new QSpinBox();
	focusEnd->setRange(0,255);
	focusEnd->setValue(255);

	animButton = new QPushButton("Start");
	connect(animButton, SIGNAL(clicked()), SLOT(startAnimation()));

	QGridLayout *mainLayout = new QGridLayout();
	mainLayout->addWidget(framesLabel,0,0);
	mainLayout->addWidget(frames,0,1);
	mainLayout->addWidget(focusStartLabel,1,0);
	mainLayout->addWidget(focusStart,1,1);
	mainLayout->addWidget(focusEndLabel,2,0);
	mainLayout->addWidget(focusEnd,2,1);
	mainLayout->addWidget(animButton,3,1);
	setLayout(mainLayout);
}

private slots:

//-----------------------------------------------------------------
// Summary: renders frame sequence and writes out images to files.
//-----------------------------------------------------------------
void startAnimation() {
//-----------------------------------------------------------------
	clock_t start = clock();
	// enable interpolation
	lensBlurApp->parameters.setInterpolation(1);

	// select folder to write frames
	// source image
	QString animDirPath = QFileDialog::getExistingDirectory(this, tr("Select animation folder"), QDir::currentPath());
	if (!animDirPath.isEmpty()) {
		float focusStep = (float)(focusEnd->value() - focusStart->value()) / (float)(frames->value()-1);
		for(int i=0;i<frames->value();i++) {
			char text[64];
			sprintf_s(text,64,"Render frame %d / %d",(i+1),frames->value());
			imagePanel->showMessage(QString(text));
			// animate focus
			int focus = focusStart->value() + (int)((float)i*focusStep);
			lensBlurApp->parameters.setFocus(focus);
			// render frame
			lensBlurApp->render();
			// draw frame
			imagePanel->drawImage(lensBlurApp->resultImage);
			imagePanel->update();
			// save image
			char fileName[128];
			sprintf_s(fileName,128,"/frame%02d.png",i);
			QString animFilePath(animDirPath);
			animFilePath.append(fileName);
			imagePanel->saveImage(animFilePath);
		}
	}

	// write out calculation time
	double animTime = (double)(clock() - start) / CLOCKS_PER_SEC;
	char msg[64];
	sprintf_s(msg,64,"Animation time: %f sec",animTime);
	imagePanel->showMessage(QString(msg));

	// disable interpolation
	lensBlurApp->parameters.setInterpolation(0);
}

};

//===============================================================
// Description: handler of the 'GPU' tab.
//===============================================================
class GPUParametersPanel : public QWidget
//===============================================================
{
	Q_OBJECT
private: 
	LensBlurApp*	lensBlurApp;
	Parameters*		parameters;

	QLabel*			deviceTypeLabel;
	QComboBox*		deviceType;
	QLabel*			blockSizeLabel;
	QSpinBox*		blockSize;
	QLabel*			threadSizeLabel;
	QSpinBox*		threadSize;

//-----------------------------------------------------------------
// Summary: initializes the engine by the selected the calculation mode (CPU / GPU).
//-----------------------------------------------------------------
void initByDevice(int deviceIndex) {
//-----------------------------------------------------------------
	// CPU
	if (deviceIndex == 0) {
		//blockSize->setEnabled(false);
		threadSize->setEnabled(false);

		lensBlurApp->imageBuffer->useGPU = false;
	}
	// GPU
	else {
		//blockSize->setEnabled(true);
		threadSize->setEnabled(true);

		lensBlurApp->imageBuffer->useGPU = true;
		// reinit gpu
		lensBlurApp->imageBuffer->uninitGPU();
		lensBlurApp->imageBuffer->initGPU(*lensBlurApp->image,*lensBlurApp->zMap,lensBlurApp->apKernel);
	}
}

public:

//-----------------------------------------------------------------
// Summary: Constructs the kernel parameter handler.
// Arguments: lensBlurApp - application controller
//-----------------------------------------------------------------
GPUParametersPanel(LensBlurApp* lensBlurApp) {
//-----------------------------------------------------------------
	this->lensBlurApp = lensBlurApp;
	this->parameters = &(lensBlurApp->parameters);

	deviceTypeLabel = new QLabel("Device type:");
	deviceType = new QComboBox();
	deviceType->addItem(QString("No device (use CPU)"));
	char** deviceNames = lensBlurApp->getGPUDevices();
	for(int i=0;deviceNames && deviceNames[i];i++) {
		deviceType->addItem(QString(deviceNames[i]));
	}
	connect(deviceType, SIGNAL(currentIndexChanged(int)), SLOT(deviceChanged()));

	//blockSizeLabel = new QLabel("Block size:");
	//blockSize = new QSpinBox();
	//blockSize->setRange(0,16);
	//blockSize->setValue(parameters->getGPUBlocks());

	threadSizeLabel = new QLabel("Thread size:");
	threadSize = new QSpinBox();
	threadSize->setRange(0,16);
	threadSize->setValue(parameters->getGPUThreads());

	QGridLayout *mainLayout = new QGridLayout();
	//mainLayout->setMenuBar(menuBar);
	mainLayout->addWidget(deviceTypeLabel,0,0);
	mainLayout->addWidget(deviceType,0,1);
	//mainLayout->addWidget(blockSizeLabel,1,0);
	//mainLayout->addWidget(blockSize,1,1);
	mainLayout->addWidget(threadSizeLabel,1,0);
	mainLayout->addWidget(threadSize,1,1);
	setLayout(mainLayout);

	initByDevice(deviceType->currentIndex());
}

private slots:

//-----------------------------------------------------------------
// Summary: selects new device for GPU calculation.
//-----------------------------------------------------------------
void deviceChanged() {
//-----------------------------------------------------------------
	int deviceIndex = deviceType->currentIndex();
	initByDevice(deviceIndex);
}

};

//===============================================================
// Description: tab panel for setting parameters.
//===============================================================
class ParameterTab : public QTabWidget
//===============================================================
{
private:
	BasicParametersPanel* basicParameters;
	AnimParametersPanel* animParameters;
	GPUParametersPanel* gpuParameters;
public:

//-----------------------------------------------------------------
// Summary: Constructs the parameter tab.
// Arguments: lensBlurApp - application controller
//            imagePanel - image panel
//-----------------------------------------------------------------
ParameterTab(LensBlurApp* lensBlurApp,ImagePanel* imagePanel) {
//-----------------------------------------------------------------
	basicParameters = new BasicParametersPanel(lensBlurApp,imagePanel);
	animParameters = new AnimParametersPanel(lensBlurApp,imagePanel);
	gpuParameters = new GPUParametersPanel(lensBlurApp);
	addTab(basicParameters,QString("Basic"));
	addTab(animParameters,QString("Animation"));
	addTab(gpuParameters,QString("GPU"));
}

//-----------------------------------------------------------------
// Returns: the focus widget from the basic parameters tab.
//-----------------------------------------------------------------
QSpinBox* getFocusWidget() {
//-----------------------------------------------------------------
	return basicParameters->getFocusWidget();
}

};

#endif
