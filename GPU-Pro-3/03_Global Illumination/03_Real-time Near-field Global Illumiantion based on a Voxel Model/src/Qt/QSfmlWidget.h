#ifndef QSFMLWIDGET_H
#define QSFMLWIDGET_H

// STL
#include <iostream>
#include <string>

// Qt
#include <QObject>
#include <QWidget>
#include <QTimer>
#include <QEvent>

// SFML
#include "SFML/Window.hpp"

using namespace std;

///
/// QSfmlWidget allows to run SFML in a Qt control. 
///

class QSfmlWidget : public QWidget, public sf::Window
{
   Q_OBJECT

public:

   ///
   /// Constructor. 
   /// Implementation is based on a SFML sample file.
   /// Intended as a base class that should be derived from.
   /// \param contextSettings Defines the settings of the OpenGL
   ///        context attached to a SFML window
   /// \param timerInterval Update timer interval in ms
   /// \param parent Parent of the widget
   ///
   QSfmlWidget(sf::ContextSettings contextSettings, int timerInterval = 0, QWidget* parent = 0);

   ///
   /// Destructor
   ///
   virtual ~QSfmlWidget();


private:

   ///
   /// Notification for the derived class that moment is good
   /// for doing initializations
   ///
   virtual void initialize();

   ///
   /// Notification for the derived class that moment is good
   /// for doing its update and drawing stuff
   ///
   virtual void refresh();


   ///
   /// Reimplements QWidget::paintEngine().
   /// Return the paint engine used by the widget to draw itself.
   /// To render outside of Qt's paint system you need to reimplement
   /// QWidget::paintEngine() to return 0.
   /// \return 0
   ///
   virtual QPaintEngine* paintEngine() const;


   /// Reimplements QWidget::event(QEvent*).
   /// This is the main event handler.
   /// Called each time an event is received by the widget ;
   /// we use it to catch the Polish event and initialize
   /// our SFML window.
   ///
   /// \param event : Event's attributes
   ///
   virtual bool event(QEvent* event);


   ///
   /// Reimplements QWidget's paintEvent 
   /// to receive paint events passed in event.
   /// Each widget performs all painting operations 
   /// from within its paintEvent() function. 
   /// Called when the widget needs to be painted,
   /// e.g. when repaint() or update() was invoked.
   /// \param event Paint events are sent to widgets 
   ///              that need to update themselves
   ///
   virtual void paintEvent(QPaintEvent*);


   ////////////////////////////////////////////////////////////

   // Member data

   QTimer updateTimer; ///< Timer used to update the view
   sf::ContextSettings contextSettings;


};


#endif
