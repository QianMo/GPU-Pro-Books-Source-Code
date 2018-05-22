#include "QSfmlWidget.h"


QSfmlWidget::QSfmlWidget(sf::ContextSettings contextSettings, int timerInterval, QWidget* parent)
: QWidget(parent)
{
   this->contextSettings = contextSettings;

   // Setup some states to allow direct rendering into the widget

   // Indicates that the widget wants to draw 
   // directly onto the screen. 
   setAttribute(Qt::WA_PaintOnScreen);

   // Indicates that the widget paints all its pixels 
   // when it receives a paint event.
   setAttribute(Qt::WA_OpaquePaintEvent);

   // Indicates that the widget has no background,
   // i.e. when the widget receives paint events, 
   // the background is not automatically repainted.
   setAttribute(Qt::WA_NoSystemBackground);


   // Set strong focus to enable keyboard events to be received
   setFocusPolicy(Qt::StrongFocus);

   // Setup the timer
   updateTimer.setInterval(timerInterval);

}


QSfmlWidget::~QSfmlWidget()
{
    // Nothing to do by default
}


void QSfmlWidget::initialize()
{
    // Nothing to do by default
}


void QSfmlWidget::refresh()
{
   // Nothing to do by default
}


QPaintEngine* QSfmlWidget::paintEngine() const
{
   return 0;
}

	
bool QSfmlWidget::event(QEvent* event)
{
   if (event->type() == QEvent::Polish)
   {
      // Handle Polish event when widget has been fully constructed
      // but before it is shown the very first time

      // sf::Window::Create creates the SFML window with the widget handle.
      // Use this function if you want to create an OpenGL
      // rendering area into an already existing control.
      // QWidget::winId() returns the window system identifier of the widget.

      Create(winId(), contextSettings);

      // Let the derived class do its specific stuff
      initialize();

      // Setup the timer to trigger a refresh at specified framerate
      // repaint() calls QWidget::paintEvent()
      connect(&updateTimer, SIGNAL(timeout()), this, SLOT(repaint()));
      updateTimer.start();
   }

   return QWidget::event(event);

}


void QSfmlWidget::paintEvent(QPaintEvent*)
{
    // Let the derived class do its specific stuff
    refresh();

    // Display on screen (swap buffers)
    // This is the sf::Window::Display function
    Display(); 
}
