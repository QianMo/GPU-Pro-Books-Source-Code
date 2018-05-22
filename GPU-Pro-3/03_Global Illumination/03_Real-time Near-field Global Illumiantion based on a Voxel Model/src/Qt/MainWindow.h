#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// Qt
#include <QMainWindow>
#include <QtGui>
#include <QXmlStreamWriter>

class CentralWidget;
class SfmlView;

///
/// MainWindow contains all GUI elements used for parameter manipulation.
///
class MainWindow : public QMainWindow
{
   Q_OBJECT

public:
   MainWindow(SfmlView* sfmlView);

   public slots:
      void getScreenshotFileName();
      void saveScreenshot();
      void saveSceneXML();
      void loadEnvMap();
      void loadParameterXML();
      void saveParameterXML();
      void setFPS(float fps);
      void toggleSwitchOffAction(bool checked);
      void showAbout();

protected:
   ///
   /// Reimplements QWidget::closeEvent, forces all 
   /// top-level widgets to be closed and the application to quit.
   /// closeEvent() is called when the user closes the widget
   /// (or when close() is called).
   ///
   void closeEvent(QCloseEvent* evt);

private:
   void setupActions();
   void setupMenus();
   void setupGUI();

   // Member data

   // Actions
   QAction* exitAction;
   QAction* screenshotAction;
   QAction* saveSceneXMLAction;
   QAction* loadEnvMapAction;
   QAction* saveParameterXMLAction;
   QAction* loadParameterXMLAction;
   QAction* aboutAction;

   QAction* switchOffAction;

   QAction* viewTimerAction;
   QAction* viewStateAction;

   CentralWidget* centralWidget; ///< A main window must have a central widget.

   SfmlView* sfmlView; ///< A pointer to the OpenGL SFML Window for taking screenshots.
   QString screenshotFormat;
   QString screenshotFileName;

};

#endif
