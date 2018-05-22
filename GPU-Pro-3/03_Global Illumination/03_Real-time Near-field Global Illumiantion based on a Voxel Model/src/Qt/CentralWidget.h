#ifndef CENTRALWIDGET_H
#define CENTRALWIDGET_H

// Qt
#include <QtGui>

#include <vector>
using std::vector;

#include "Scene/SceneDataStructs.h"// defines struct Pose

class SfmlView;
class PageScene;

class CentralWidget : public QWidget
{
   Q_OBJECT
public:
   CentralWidget(const vector<Pose>& poses, int currentPoseIndex,
      SfmlView* sfmlView,
      QWidget* parent = 0);
  // ~CentralWidget();

   const QStackedWidget* getStackedWidget() const { return stack; }

   public slots:


private:
   
   QStackedWidget* stack;          ///< Provides a stack of widgets where only one widget is visible at a time.
   QComboBox* comboStackItems;    ///< Used for selecting the current visible widget of stack.

   QTabWidget* tabs;
};

#endif
