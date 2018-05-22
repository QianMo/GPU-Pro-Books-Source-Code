///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

// STL
#include <iostream>

// Qt
#include <QApplication>

// SFML
#include "SFML/Window.hpp"

// Own
#include "SfmlView.h"
#include "Qt/MainWindow.h"
#include "Qt/SceneXMLDocument.h"
#include "Qt/Settings.h"

using namespace std;

/// Entry point of application
///
/// \return Application exit code
///
int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

   //cout << "ScreenGeometry:" <<
   //   QApplication::desktop()->screenGeometry().left() << " " <<
   //   QApplication::desktop()->screenGeometry().right() << " " <<
   //   QApplication::desktop()->screenGeometry().bottom() << " " <<
   //   QApplication::desktop()->screenGeometry().top() << " " << endl;


   // Query for custom XML load file
   QString initialPath = QDir::currentPath() + "/SceneXML/";

   QString filename = QFileDialog::getOpenFileName(0, ("Open Scene XML Description"),
      initialPath,
      QString("%1 Files (*.%2);;All Files (*)")
      .arg("XML")
      .arg("xml"));

   SceneData* data = 0;
   if (!filename.isEmpty() ) 
   {
      data = SceneXMLDocument::getDataFromFile(filename);
   }
   else
   {
      std::cout << "No Scene XML file selected. Exit." << std::endl;
      return -1;
   }


   // Timer Query Results 

   QTextEdit timerMonitor("Timer");
   //timerResults.setWindowFlags(Qt::Tool);
   timerMonitor.setWindowTitle("Timings [ms]");
   timerMonitor.setReadOnly( true );

	// Create a SFML view

   // Use OpenGL 2 Version 2.1

	sf::ContextSettings contextSettings;
	contextSettings.MajorVersion = 2;
	contextSettings.MinorVersion = 1;
	contextSettings.DepthBits    = 24;
	contextSettings.StencilBits  = 8;
	contextSettings.AntialiasingLevel = 8; 

   // Create Main Window
   SfmlView* sfmlView = new SfmlView(contextSettings, data, 0, 0, &timerMonitor);
	sfmlView->show();

	MainWindow settings(sfmlView);
	settings.show();
   timerMonitor.show();

	sfmlView->SetPosition(10, 50);
   int padding = 25;
   settings.setGeometry(min(QApplication::desktop()->screenGeometry().right()-260, sfmlView->geometry().right() + padding),
      sfmlView->geometry().top(),
      0, 0);
   timerMonitor.setGeometry(min(QApplication::desktop()->screenGeometry().right()-260, sfmlView->geometry().right() + padding),
      settings.geometry().bottom() + padding*1.5,
      260, 150);

	return app.exec();
}
