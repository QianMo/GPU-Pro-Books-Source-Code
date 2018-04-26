/* ******************************************************************************
* Description: Main window of the application. Contains a panel for displaying 
*              the dof image, and a panel where the user can set the render parameters.
*
*  Version 1.0.0
*  Date: Jul 16, 2009
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _MAINWINDOW_
#define _MAINWINDOW_

#include <QtGui/QAction>
#include <QtGui/QActionGroup>
#include <QtGui/QLabel>
#include <QtGui/QTextEdit>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QStatusBar>
#include <QtGui/QContextMenuEvent>

#include <QMainWindow.h>

#include "qt/imagepanel.h"
#include "qt/parametertab.h"
#include "lensBlurApp.hpp"

static const char* FILE_FILTER = "Images (*.png);;All Files (*.*)";

//===============================================================
class AboutDialog : public QDialog
//===============================================================
{
	Q_OBJECT
private:
	QTextEdit*			message;
	QPushButton*	okButton;
public:
	AboutDialog() {
		message = new QTextEdit();
		QString text = QString("The application demonstrates a realistic Depth-of-Field algorithm running on CPU or GPU device, based on local neighborhood blending together with a unique edge improvement technique.\n\n");
		text.append("Created by: David Illes, Peter Horvath\nwww.postpipe.hu\n\n");
		text.append("-------------------------------------------------------------------------------------\n\n");
		text.append("Input: source image\n           greyscaled depth map image\nOutput: DoF image\n\n");
		text.append("Parameters:\n");
		text.append(" - Basic parameters: DoF and render settings.\n");
		text.append("   * Mode: defines the calculation method.\n");
		text.append("     - Artist: user can define blur parameters.\n");
		text.append("       * Blur strength: defines how much the far pixels are blurred.\n");
		text.append("     - Physic: the blur size computed from lens parameters.\n");
		text.append("       * fLength: focal length of the lens.\n");
		text.append("       * ms: subject magnification.\n");
		text.append("       * fStop: the focal length divided by the diameter of the lens.\n");
		text.append("       * Image range: the distance between the front and near clip of the image (in meter).\n");
		text.append("       * Camera distance: the disance of the camera from the front clip (in meter).\n");
		text.append("   * Z epsilon: defines the edges on the image. When the difference of two neighboring pixels' z value is under the zEpsilon limit then they come from different objects.\n");
		text.append("   * Smoothing: blend width on the edge improvement.\n");
		text.append("   * Mask: type of the kernel mask which contains weights for neighbor blending. You can select from 7 defined kernel type.\n");
		text.append("   * Focal point: z value of the pixel which is in the focus.\n");
		text.append("   * Bloom amount: strength of the bloom effect.\n");
		text.append("   * Bloom threshold: defines a luminance limit where the bloom effect is applied to.\n");
		text.append("\n - Animation parameters: writes out frame sequence into a user defined directory with animated focal point.\n");
		text.append("   * Number of frames: number of frames in the animation sequence\n");
		text.append("   * Focus at first frame: z value of the pixel which is in the focus on the first frame\n");
		text.append("   * Focus at last frame: z value of the pixel which is in the focus on the last frame\n");
		text.append("\n - GPU parameters: select device for calculation.\n");
		text.append("   * Device type: target of the calculation (CPU or GPU)\n");
		text.append("   * Thread size: number of GPU threads. A GPU thread computes one pixel of the image. The block count is defined from the number of pixels. Be careful with this parameter, high thread number can cause crash.\n");
		message->setText(text);
		message->setReadOnly(true);
		okButton = new QPushButton("OK");
		connect(okButton, SIGNAL(clicked()), SLOT(close()));

		QBoxLayout *mainLayout = new QBoxLayout(QBoxLayout::Direction::TopToBottom);
		mainLayout->addWidget(message,0);
		mainLayout->addWidget(okButton,0);
		setLayout(mainLayout);

		setWindowTitle(tr("About DoF Demo"));
		setMinimumWidth(450);
		setMinimumHeight(400);
	}
};

//===============================================================
class MainWindow : public QMainWindow
//===============================================================
{
	Q_OBJECT

public: 
	ImagePanel*		imagePanel;
private:
	LensBlurApp*	blurEngine;
	ParameterTab*	parameterTab;

	QString			saveFilePath;

    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *viewMenu;
    QMenu *helpMenu;
    QActionGroup *alignmentGroup;
    QAction *newAct;
    QAction *openSourceAct;
	QAction *openZMapAct;
    QAction *saveAct;
	QAction *saveAsAct;
	QAction *showSourceAct;
	QAction *showZmapAct;
    QAction *exitAct;
    QAction *aboutAct;
    QAction *aboutQtAct;

public:

//-----------------------------------------------------------------
// Summary: Constructs the application window.
// Arguments: lensBlurApp - application controller
//-----------------------------------------------------------------
MainWindow(LensBlurApp* blurEngine) {
//-----------------------------------------------------------------
	this->blurEngine = blurEngine;

	QWidget *widget = new QWidget;
	setCentralWidget(widget);

	imagePanel = new ImagePanel(blurEngine,statusBar());
	parameterTab = new ParameterTab(blurEngine,imagePanel);
	parameterTab->setMaximumWidth(250);
	imagePanel->setFocusWidget(parameterTab->getFocusWidget());

	QGridLayout *layout = new QGridLayout();
	layout->setMargin(5);
	layout->addWidget(imagePanel,0,0);
	layout->addWidget(parameterTab,0,1);
	widget->setLayout(layout);

	createActions();
	createMenus();

	QString message = tr("A context menu is available by right-clicking");
	statusBar()->showMessage(message);

	setWindowTitle(tr("Depth of Field DEMO"));
	setMinimumSize(300, 300);
	resize(1000, 650);
}

//-----------------------------------------------------------------
// Summary: creates an image from the inner image representation.
// Arguments: image - inner image format
//            qimage - displayable image
//-----------------------------------------------------------------
void loadImage(Image** image, QImage& qimage) {
//-----------------------------------------------------------------
	if (*image) delete *image;
	*image = new Image(qimage.width(),qimage.height(),qimage.depth());
	for(int y=0;y<qimage.height();y++) {
		for(int x=0;x<qimage.width();x++) {
			int index = y*qimage.width() + x;
			QRgb color = qimage.pixel(x,y);
			(*image)->pixels[index].r = (float)(color>>16 & 0x000000FF);
			(*image)->pixels[index].g = (float)(color>>8 & 0x000000FF);
			(*image)->pixels[index].b = (float)(color & 0x000000FF);
			// caculate luminance
			(*image)->pixels[index].luminance();	
		}
	}
}

//-----------------------------------------------------------------
// Summary: loads the source and z map images given by the user on the GUI.
//-----------------------------------------------------------------
void loadSourceAndDepth() {
//-----------------------------------------------------------------
	// source image
	QString sourceFileName = QFileDialog::getOpenFileName(this, tr("Open source image"), QDir::currentPath().append("/images"), tr(FILE_FILTER));
	if (!sourceFileName.isEmpty()) {
		QImage sourceImage(sourceFileName);
		if (sourceImage.isNull()) {
			QMessageBox::information(this, tr("Source image"),tr("Cannot load %1.").arg(sourceFileName));
			return;
		}

		// load the new image
		loadImage(&blurEngine->image,sourceImage);

		// depth map
		QString zFileName = QFileDialog::getOpenFileName(this, tr("Open depth image"), QDir::currentPath().append("/images"), tr(FILE_FILTER));
		if (!zFileName.isEmpty()) {
			QImage zImage(zFileName);
			if (zImage.isNull()) {
				QMessageBox::information(this, tr("Depth image"),tr("Cannot load %1.").arg(zFileName));
				return;
			}

			// load the new image
			loadImage(&blurEngine->zMap,zImage);
			blurEngine->initZMap(blurEngine->zMap);

			if (blurEngine->resultImage) delete blurEngine->resultImage;
			blurEngine->resultImage = new Image(blurEngine->image->getWidth(),blurEngine->image->getHeight(),24);

			recreateImageBuffer();

			imagePanel->rerenderImage();
		}
	}
}

private:

//-----------------------------------------------------------------
// Summary: creates actions for the menu.
//-----------------------------------------------------------------
void createActions() {
//-----------------------------------------------------------------
	newAct = new QAction(tr("&New"), this);
	newAct->setShortcuts(QKeySequence::New);
	newAct->setStatusTip(tr("Create a DOF image"));
	connect(newAct, SIGNAL(triggered()), this, SLOT(newImage()));

    openSourceAct = new QAction(tr("&Open image..."), this);
	openSourceAct->setShortcuts(QKeySequence::Open);
	openSourceAct->setStatusTip(tr("Open image"));
	connect(openSourceAct, SIGNAL(triggered()), this, SLOT(openSource()));

	openZMapAct = new QAction(tr("&Open z map..."), this);
	openZMapAct->setShortcuts(QKeySequence::Open);
	openZMapAct->setStatusTip(tr("Open depth map"));
	connect(openZMapAct, SIGNAL(triggered()), this, SLOT(openZMap()));

	saveAct = new QAction(tr("&Save"), this);
	saveAct->setShortcuts(QKeySequence::Save);
	saveAct->setStatusTip(tr("Save image to disk"));
	connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

	saveAsAct = new QAction(tr("&Save as..."), this);
	saveAsAct->setShortcuts(QKeySequence::Save);
	saveAsAct->setStatusTip(tr("Save image to disk"));
	connect(saveAsAct, SIGNAL(triggered()), this, SLOT(saveAs()));

	showSourceAct = new QAction(tr("View &source"), this);
	showSourceAct->setShortcut(tr("F2"));
	showSourceAct->setStatusTip(tr("Show source image"));
	connect(showSourceAct, SIGNAL(triggered()), this, SLOT(showSource()));

	showZmapAct = new QAction(tr("View &z map"), this);
	showZmapAct->setShortcut(tr("F3"));
	showZmapAct->setStatusTip(tr("Show depth map image"));
	connect(showZmapAct, SIGNAL(triggered()), this, SLOT(showZmap()));

    exitAct = new QAction(tr("E&xit"), this);
	exitAct->setShortcut(tr("Ctrl+Q"));
	exitAct->setStatusTip(tr("Exit the application"));
	connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

    aboutAct = new QAction(tr("A&bout"), this);
	aboutAct->setShortcut(tr("F1"));
	aboutAct->setStatusTip(tr("Information about the application"));
	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));
}

//-----------------------------------------------------------------
// Summary: creates the menu.
//-----------------------------------------------------------------
void createMenus() {
//-----------------------------------------------------------------
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(newAct);
	fileMenu->addAction(openSourceAct);
	fileMenu->addAction(openZMapAct);
	fileMenu->addAction(saveAct);
	fileMenu->addAction(saveAsAct);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);

	viewMenu = menuBar()->addMenu(tr("&View"));
	viewMenu->addAction(showSourceAct);
	viewMenu->addAction(showZmapAct);

    helpMenu = menuBar()->addMenu(tr("&Help"));
	helpMenu->addAction(aboutAct);
}

//-----------------------------------------------------------------
// Summary: creates new image buffer keeping the current settings.
//-----------------------------------------------------------------
void recreateImageBuffer() {
//-----------------------------------------------------------------
	bool useGPU = blurEngine->imageBuffer->useGPU;
	if (blurEngine->imageBuffer) delete blurEngine->imageBuffer;
	blurEngine->imageBuffer = new ImageBuffer(blurEngine->image,blurEngine->zMap,blurEngine->apKernel,blurEngine->resultImage,&(blurEngine->parameters));
	blurEngine->imageBuffer->useGPU = useGPU;

	// reinit gpu
	if (useGPU) {
		blurEngine->imageBuffer->uninitGPU();
		blurEngine->imageBuffer->initGPU(*blurEngine->image,*blurEngine->zMap,blurEngine->apKernel);
	}
}

private slots:

//-----------------------------------------------------------------
// Summary: loads new source and z map image.
//-----------------------------------------------------------------
void newImage() {
//-----------------------------------------------------------------
	loadSourceAndDepth();
}

//-----------------------------------------------------------------
// Summary: loads new source image.
//-----------------------------------------------------------------
void openSource() {
//-----------------------------------------------------------------
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open source image"), QDir::currentPath().append("/images"), tr(FILE_FILTER));
	if (!fileName.isEmpty()) {
		QImage image(fileName);
		if (image.isNull()) {
			QMessageBox::information(this, tr("Source image"),tr("Cannot load %1.").arg(fileName));
			return;
		}

		// load the new image
		loadImage(&blurEngine->image,image);

		if (blurEngine->resultImage) delete blurEngine->resultImage;
		blurEngine->resultImage = new Image(blurEngine->image->getWidth(),blurEngine->image->getHeight(),24);

		recreateImageBuffer();

		imagePanel->drawImage(blurEngine->image);
	}
}

//-----------------------------------------------------------------
// Summary: loads new z map image.
//-----------------------------------------------------------------
void openZMap() {
//-----------------------------------------------------------------
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open depth image"), QDir::currentPath().append("/images"), tr(FILE_FILTER));
	if (!fileName.isEmpty()) {
		QImage image(fileName);
		if (image.isNull()) {
			QMessageBox::information(this, tr("Depth image"),tr("Cannot load %1.").arg(fileName));
			return;
		}

		// load the new image
		loadImage(&blurEngine->zMap,image);
		blurEngine->initZMap(blurEngine->zMap);

		recreateImageBuffer();
	}
}

//-----------------------------------------------------------------
// Summary: saves the result image. Overwrites the last saved file.
//-----------------------------------------------------------------
void save() {
//-----------------------------------------------------------------
	if (!saveFilePath.isEmpty()) {
		imagePanel->saveImage(saveFilePath);
	} else {
		saveAs();
	}
}

//-----------------------------------------------------------------
// Summary: saves the result image to a new file.
//-----------------------------------------------------------------
void saveAs() {
//-----------------------------------------------------------------
	saveFilePath = QFileDialog::getSaveFileName(this, tr("Save image"), QDir::currentPath(),tr(FILE_FILTER));
	if (!saveFilePath.isEmpty()) {
		imagePanel->saveImage(saveFilePath);
	}
}

//-----------------------------------------------------------------
// Summary: about the application.
//-----------------------------------------------------------------
void about() {
//-----------------------------------------------------------------
	AboutDialog aboutDialog;
	aboutDialog.exec();
}

//-----------------------------------------------------------------
// Summary: displays the source image on the image panel.
//-----------------------------------------------------------------
void showSource() {
//-----------------------------------------------------------------
	imagePanel->drawImage(blurEngine->image);
}

//-----------------------------------------------------------------
// Summary: displays the z map image on the image panel.
//-----------------------------------------------------------------
void showZmap() {
//-----------------------------------------------------------------
	imagePanel->drawImage(blurEngine->zMap);
}

};

#endif

