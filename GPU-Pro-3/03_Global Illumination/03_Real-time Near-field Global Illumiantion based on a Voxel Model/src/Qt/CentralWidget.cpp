#include "CentralWidget.h"

#include "Qt/PageDebugging.h"
#include "Qt/PageFilter.h"
#include "Qt/PageIndirectLight.h"
#include "Qt/PageScene.h"
#include "Qt/PageToneMapping.h"
#include "Qt/Settings.h"

#include "Common.h"
#include "Scene/Scene.h"
#include "SfmlView.h"

CentralWidget::CentralWidget(const vector<Pose>& poses,
                             int currentPoseIndex,
                              SfmlView* sfmlView,
                              QWidget* parent) : QWidget(parent)
{
   // Create and add elements to stacked widget 

   stack = new QStackedWidget();
   PageScene* sceneWidget = new PageScene(poses, currentPoseIndex, stack);
   stack->addWidget(sceneWidget);

   connect(sfmlView, SIGNAL(sceneLoaded()), sceneWidget, SLOT(addSceneElementsList()));

   stack->addWidget(new PageIndirectLight(Scene::Instance()->getWindowWidth(), Scene::Instance()->getWindowHeight(), stack));
   stack->addWidget(new PageFilter(stack));
   stack->addWidget(new PageToneMapping(stack));
   stack->addWidget(new PageDebugging(stack));

   // create a combo box and connect it to the stacked widget
   comboStackItems = new QComboBox();
   comboStackItems->setStyleSheet("* { background-color: rgb(180,220,255); }");
   comboStackItems->view()->setStyleSheet("* { background-color: rgb(255,255,255); }");

   comboStackItems->addItem("Scene");
   comboStackItems->addItem("Indirect Light");
   comboStackItems->addItem("Filter");
   comboStackItems->addItem("Tone Mapping");
   comboStackItems->addItem("Visualize Voxels");
   
   connect(comboStackItems, SIGNAL(activated(int)), stack, SLOT(setCurrentIndex(int)));

   // Central Widget's layout
   QVBoxLayout* layout = new QVBoxLayout();
   layout->addWidget(comboStackItems);
   layout->addWidget(stack);
   setLayout(layout);

   comboStackItems->setCurrentIndex(1);
   stack->setCurrentIndex(1);

}
