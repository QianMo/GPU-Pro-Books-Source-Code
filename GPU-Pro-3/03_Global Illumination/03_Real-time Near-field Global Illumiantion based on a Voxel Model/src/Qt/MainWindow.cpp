#include "MainWindow.h"


#include "Qt/CentralWidget.h"
#include "Qt/SceneXMLDocument.h"

#include "Common.h"

#include "Lighting/EnvMap.h"
#include "Scene/Scene.h"
#include "SfmlView.h"


MainWindow::MainWindow(SfmlView* sfmlView) : sfmlView(sfmlView)
{
   setupGUI();
   setupActions();
   setupMenus();

   // Appearance settings

   QStyle* myStyle = new QPlastiqueStyle();
   setStyle(myStyle);
   //setWindowFlags(Qt::Tool);

   QList<QWidget*> widgets = qFindChildren<QWidget*>(this);
   foreach (QWidget* w, widgets)
   {
      w->setStyle(myStyle);
   }

   connect(sfmlView, SIGNAL(updatedFPS(float)), this, SLOT(setFPS(float)));
   setMinimumSize(260, 700);

}

void MainWindow::setupActions()
{
   // Create Actions
   exitAction = new QAction("Exit",this);	
   exitAction->setShortcut(tr("Ctrl+X"));

   screenshotAction = new QAction("Save Screenshot", this);
   screenshotAction->setShortcut(tr("Ctrl+S"));

   saveSceneXMLAction = new QAction("Save Scene as XML", this);
   
   saveParameterXMLAction = new QAction("Save Parameters as XML", this);
   loadParameterXMLAction = new QAction("Load Parameters from XML", this);

   loadEnvMapAction = new QAction("Load Env Map", this);

   aboutAction = new QAction("About", this);
   connect(aboutAction, SIGNAL(triggered()), this, SLOT(showAbout()));

   screenshotFormat = "png";

   switchOffAction = new QAction("Switch On/Off", this);
   switchOffAction->setCheckable(true);
   switchOffAction->setChecked(true);

   // Connect Actions
   connect(exitAction, SIGNAL(triggered()), this, SLOT(close()));
   connect(screenshotAction, SIGNAL(triggered()), this, SLOT(getScreenshotFileName()));
   connect(saveSceneXMLAction, SIGNAL(triggered()), this, SLOT(saveSceneXML()));
   connect(loadEnvMapAction, SIGNAL(triggered()), this, SLOT(loadEnvMap()));
   connect(saveParameterXMLAction, SIGNAL(triggered()), this, SLOT(saveParameterXML()));
   connect(loadParameterXMLAction, SIGNAL(triggered()), this, SLOT(loadParameterXML()));

   viewTimerAction = new QAction("Timer Query Results", this);
  
   connect(viewTimerAction, SIGNAL(toggled(bool)), sfmlView, SLOT(toggleTimerMonitor(bool)));
 
   viewTimerAction->setCheckable(true);
   viewTimerAction->setChecked(true);
 
   int pageIndirectLightIndex = 1;
   int pageFilterIndex = 2;
   
   connect(switchOffAction, SIGNAL(toggled(bool)), centralWidget->getStackedWidget()->widget(pageIndirectLightIndex), SLOT(toggleIndirectLightCheckBox(bool)));
   connect(centralWidget->getStackedWidget()->widget(pageIndirectLightIndex), SIGNAL(toggledIndirectLightCheckBox(bool)), this, SLOT(toggleSwitchOffAction(bool)));

}

void MainWindow::toggleSwitchOffAction(bool checked)
{
   if(switchOffAction->isChecked() != checked)
   {
      switchOffAction->setChecked(checked);
   }
}

void MainWindow::setupMenus()
{
   // Create Menu

   QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
   fileMenu->addAction(screenshotAction);
   fileMenu->addAction(saveSceneXMLAction);
   fileMenu->addAction(loadEnvMapAction);
   fileMenu->addAction(exitAction);

   QMenu* paramMenu = menuBar()->addMenu(tr("&Parameter Sets"));
   paramMenu->addAction(loadParameterXMLAction);
   paramMenu->addAction(saveParameterXMLAction);
   paramMenu->addSeparator();
   paramMenu->addAction(switchOffAction);

   QMenu* viewMenu = menuBar()->addMenu("View");
   viewMenu->addAction(viewTimerAction);

   menuBar()->addAction(aboutAction);

}

void MainWindow::setupGUI()
{
   // Central Widget
   centralWidget = new CentralWidget(Scene::Instance()->getCameraPoses(),
      Scene::Instance()->getCurrentCameraPoseIndex(),
      sfmlView, this);
   setCentralWidget(centralWidget);

   statusBar()->setSizeGripEnabled(false);

   setWindowTitle("Settings");
}


void MainWindow::closeEvent(QCloseEvent* evt)
{
   Q_UNUSED( evt );

   foreach (QWidget* widget, QApplication::topLevelWidgets())
   {
      widget->close();
   }

   QApplication::quit();
}

void MainWindow::showAbout()
{
   QMessageBox::about(this, "VGI-Demo",
      "Real-time Near-Field Global Illumination based on a Voxel Model\n"\
      "GPU Pro 3\n"\
      "Authors: Sinje Thiedemann, Niklas Henrich, Thorsten Grosch, Stefan Mueller"
      );
}

void MainWindow::getScreenshotFileName()
{
   string f = "untitled";

   QString stamp = QDate::currentDate().toString("ddMMMyyyy_")+QTime::currentTime().toString("hhmmss");
   QString initialPath = QDir::currentPath() + "/Screenshots/" + stamp + "_" + QString::fromStdString(f) + "." +screenshotFormat;

   screenshotFileName = QFileDialog::getSaveFileName(this, tr("Save As"),
      initialPath,
      tr("%1 Files (*.%2);;All Files (*)")
      .arg(screenshotFormat.toUpper())
      .arg(screenshotFormat));


   if (!screenshotFileName.isEmpty())
   {
      //QTimer::singleShot(1000, this, SLOT(saveScreenshot()));
      sfmlView->saveScreenshotNextFrame(screenshotFileName.toStdString());
   }
}

void MainWindow::saveScreenshot()
{
   QPixmap::grabWindow(sfmlView->winId()).save(screenshotFileName, screenshotFormat.toAscii());
}


void MainWindow::setFPS(float fps)
{
   statusBar()->showMessage(QString("fps: ")+QString::number(fps));
}


void MainWindow::saveSceneXML()
{
   QString stamp = QDate::currentDate().toString("_ddMMyyyy_")+QTime::currentTime().toString("hhmmss");
   QString initialPath = QDir::currentPath() + "/SceneXML/" + QString::fromStdString(SCENE->getName()) + stamp + ".xml";

   QString filename = QFileDialog::getSaveFileName(this, tr("Save As"),
      initialPath,
      tr("%1 Files (*.%2);;All Files (*)")
      .arg("XML")
      .arg("xml"));

   if ( filename.isEmpty() ) return;

   // open the file and check we can write to it
   QFile file( filename );
   if ( !file.open( QIODevice::WriteOnly ) )
   {
      QMessageBox::warning(this, "Save Scene XML", QString("Failed to write to '%1'").arg(filename));
      return;
   }

   SceneXMLDocument::saveSceneXML(file, sfmlView->GetWidth(), sfmlView->GetHeight(), sfmlView);

   //display message
   QMessageBox::information(this, "Save Scene XML", QString("Saved to '%1'").arg(filename));

}

void MainWindow::saveParameterXML()
{
   QString stamp = QDate::currentDate().toString("_ddMMyyyy_")+QTime::currentTime().toString("hhmmss");
   QString initialPath = QDir::currentPath() + "/SceneXML/ParameterSets/" + "ParameterSet" + stamp + ".xml";

   QString filename = QFileDialog::getSaveFileName(this, tr("Save Parameters As XML"),
      initialPath,
      tr("%1 Files (*.%2);;All Files (*)")
      .arg("XML")
      .arg("xml"));

   if ( filename.isEmpty() ) return;

   // open the file and check we can write to it
   QFile file( filename );
   if ( !file.open( QIODevice::WriteOnly ) )
   {
      QMessageBox::warning(this, "Save Parameter XML", QString("Failed to write to '%1'").arg(filename));
      return;
   }

   SceneXMLDocument::saveParameterXML(file, sfmlView);

   //display message
   QMessageBox::information(this, "Save Parameter XML", QString("Saved to '%1'").arg(filename));

}

void MainWindow::loadParameterXML()
{
   QString initialPath = QDir::currentPath() + "/SceneXML/ParameterSets/";

   QString filename = QFileDialog::getOpenFileName(this, tr("Load Parameter Set"),
      initialPath,
      tr("%1 Files (*.%2);;All Files (*)")
      .arg("XML")
      .arg("xml"));

   if ( filename.isEmpty() ) return;

   SceneXMLDocument::loadParameterXML(filename, sfmlView);

}


void MainWindow::loadEnvMap()
{
   // longitude latitude, pfm

   QString initialPath = QDir::currentPath() + "/images/";

   QString filename = QFileDialog::getOpenFileName(this, tr("Load Environment Map (Longitude / Latitude)"),
      initialPath,
      tr("%1 Files (*.%2);;All Files (*)")
      .arg("PFM")
      .arg("pfm"));

   if ( filename.isEmpty() ) return;

   SCENE->getEnvMap()->loadPFM(filename.toStdString());

}