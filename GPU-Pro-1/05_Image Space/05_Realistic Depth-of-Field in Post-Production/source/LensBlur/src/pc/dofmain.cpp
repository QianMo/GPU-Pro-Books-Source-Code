/* ******************************************************************************
* Description: Depth-of-Field demo application. Input is a source image 
*              and a grey-scaled z depth map. Output is a dof image displayed
*              in a Qt-based window. The dof parameters can be set on the GUI.
*
*  Version 1.0.0
*  Date: Jul 16, 2009
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#include <stdio.h>
#include <stdlib.h>

#include <QApplication>

#include "qt/mainwindow.h"
#include "lensBlurApp.hpp"
#include "defaultKernels.hpp"

int main(int argc, char *argv[])
{
	LensBlurApp* blurEngine = new LensBlurApp();
	blurEngine->imageBuffer = new ImageBuffer();

	QApplication app(argc, argv);
    MainWindow window(blurEngine);

	// show window
    window.show();

	// initial images
	window.loadSourceAndDepth();

    return app.exec();
}