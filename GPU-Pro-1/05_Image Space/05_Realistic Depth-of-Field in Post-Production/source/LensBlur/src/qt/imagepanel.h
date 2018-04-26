/* ******************************************************************************
* Description: Scrollable panel class which displayes an image.
*
*  Version 1.0.0
*  Date: Jul 20, 2009
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _IMAGEPANEL_
#define _IMAGEPANEL_

#include <QtGui/QWidget>
#include <QtGui/QLabel>
#include <QtGui/QSpinBox>
#include <QtGui/QScrollArea>
#include <QtGui/QVBoxLayout>
#include <QtGui/QFileDialog>
#include <QtGui/QDialog>
#include <QtGui/QProgressBar>
#include <QtGui/QMessageBox>
#include <QtGui/QStatusBar>
#include <QtGui/QImage>

#include <QMainWindow.h>
#include <QMouseEvent>

#include "lensBlurApp.hpp"
#include "image.hpp"

//===============================================================
// Description: label to displaying image on it.
//===============================================================
class ImageLabel : public QLabel {
//===============================================================
private:
	LensBlurApp*	lensBlurApp;
	QStatusBar*		statusBar;

	QSpinBox*		focusWidget;
	QImage*			image;
public:

//-----------------------------------------------------------------
// Summary: Constructs an image label.
// Arguments: lensBlurApp - application controller
//            statusBar - status bar for displaying messages
//-----------------------------------------------------------------
ImageLabel(LensBlurApp* lensBlurApp, QStatusBar* statusBar) {
//-----------------------------------------------------------------
	this->lensBlurApp = lensBlurApp;
	this->statusBar = statusBar;
	this->image = 0;
	this->focusWidget = 0;
}

//-----------------------------------------------------------------
// Summary: Draws the given image.
// Arguments: resultImage - image to display
//-----------------------------------------------------------------
void drawImage(Image* resultImage) {
//-----------------------------------------------------------------
	if (image) delete image;
	image = new QImage(resultImage->width, resultImage->height, QImage::Format_RGB32);
	setFixedSize(resultImage->width,resultImage->height);
	// convert to RGB32
	int x,y;
	for(int i=0;i<resultImage->size;i++) {
		x = i%resultImage->width;
		y = i/resultImage->width;
		ImagePixel* pixel = &resultImage->pixels[i];
		int color = 0xFF000000 | (int)pixel->r<<16 | (int)pixel->g<<8 | (int)pixel->b;
		image->setPixel(x,y,color);
	}
	setPixmap(QPixmap::fromImage(*image));
}

//-----------------------------------------------------------------
// Summary: Renders a new image and displays on the screen.
//-----------------------------------------------------------------
void rerenderImage() {
//-----------------------------------------------------------------
	statusBar->showMessage(QString("Processing..."));

	lensBlurApp->render();
	drawImage(lensBlurApp->resultImage);
	// write out calculation time
	char msg[64];
	sprintf_s(msg,64,"Calculation time: %f sec",lensBlurApp->calculationTime);
	statusBar->showMessage(QString(msg));
}

//-----------------------------------------------------------------
// Returns: the currently displayed image.
//-----------------------------------------------------------------
QImage* getImage() {
//-----------------------------------------------------------------
	return image;
}

//-----------------------------------------------------------------
// Summary: sets the widget which stores the focus value.
//-----------------------------------------------------------------
void setFocusWidget(QSpinBox* widget) {
//-----------------------------------------------------------------
	focusWidget = widget;
}

//-----------------------------------------------------------------
// Summary: displays the given message on the status bar.
//-----------------------------------------------------------------
void showMessage(QString& msg) {
//-----------------------------------------------------------------
	statusBar->showMessage(msg);
}

private:

//-----------------------------------------------------------------
// Summary: mouse button release callback. Renders new image 
//          with the selected focal point.
// Arguments: ev - mouse event
//-----------------------------------------------------------------
void mouseReleaseEvent(QMouseEvent* ev) {
//-----------------------------------------------------------------
	if (ev->button() == Qt::MouseButton::LeftButton) {
		lensBlurApp->parameters.setFocus(*lensBlurApp->zMap,ev->x(),ev->y());
		rerenderImage();
		// refresh focus on the GUI
		if (focusWidget) focusWidget->setValue(lensBlurApp->parameters.getFocus());
	}
}

};

//===============================================================
// Description: panel to display an image.
//===============================================================
class ImagePanel : public QWidget
//===============================================================
{
private:
	ImageLabel* imageLabel;
    QScrollArea* scrollArea;
	QImage* image;

public:

//-----------------------------------------------------------------
// Summary: Constructs an image panel.
// Arguments: lensBlurApp - application controller
//            statusBar - status bar for displaying messages
//-----------------------------------------------------------------
ImagePanel(LensBlurApp* lensBlurApp, QStatusBar* statusBar) {
//-----------------------------------------------------------------
	imageLabel = new ImageLabel(lensBlurApp, statusBar);
	imageLabel->setBackgroundRole(QPalette::Base);
	imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	imageLabel->setScaledContents(true);

	scrollArea = new QScrollArea();
	scrollArea->setBackgroundRole(QPalette::Dark);
	scrollArea->setWidget(imageLabel);

	QVBoxLayout *mainLayout = new QVBoxLayout();
	mainLayout->addWidget(scrollArea);
	setLayout(mainLayout);

	image = 0;
}

//-----------------------------------------------------------------
// Summary: Draws the given image.
// Arguments: resultImage - image to display
//-----------------------------------------------------------------
void drawImage(Image* resultImage) {
//-----------------------------------------------------------------
	imageLabel->drawImage(resultImage);
}

//-----------------------------------------------------------------
// Summary: Renders a new image and displays on the screen.
//-----------------------------------------------------------------
void rerenderImage() {
//-----------------------------------------------------------------
	imageLabel->rerenderImage();
}

//-----------------------------------------------------------------
// Summary: redraws the image.
// Arguments: width - image width
//            height - image height
//-----------------------------------------------------------------
void refresh(int width, int height) {
//-----------------------------------------------------------------
	if (image) delete image;
	image = new QImage(width, height, QImage::Format_RGB32);
	imageLabel->setPixmap(QPixmap::fromImage(*image));
}

//-----------------------------------------------------------------
// Returns: the currently displayed image.
//-----------------------------------------------------------------
QImage* getImage() {
//-----------------------------------------------------------------
	return imageLabel->getImage();
}

//-----------------------------------------------------------------
// Summary: sets the widget which stores the focus value.
//-----------------------------------------------------------------
void setFocusWidget(QSpinBox* widget) {
//-----------------------------------------------------------------
	imageLabel->setFocusWidget(widget);
}

//-----------------------------------------------------------------
// Summary: displays the given message on the status bar.
//-----------------------------------------------------------------
void showMessage(QString& msg) {
//-----------------------------------------------------------------
	imageLabel->showMessage(msg);
}

//-----------------------------------------------------------------
// Summary: writes the currently displayed image to a file.
// Arguments: filePath - path of the image file
//-----------------------------------------------------------------
void saveImage(QString filePath) {
//-----------------------------------------------------------------
	QImage* image = imageLabel->getImage();
	image->save(filePath,"png");
}

};

#endif

