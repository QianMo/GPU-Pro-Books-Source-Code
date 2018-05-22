#include "PageScene.h"

#include "Qt/Settings.h"

#include "Common.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Scene/SpotLight.h"
#include "Scene/ObjectSequence.h"
#include "Lighting/EnvMap.h"

PageScene::PageScene(const vector<Pose>& poses, int currentPoseIndex, QWidget* parent) : QWidget(parent)
{
   layout = new QVBoxLayout();
   setLayout(layout);

   addMouseControl();
   addCameraControl(poses, currentPoseIndex);
   addLightControl();
   addModelControl();
   addGBufferControl();

   layout->addStretch(1);

}

void PageScene::addSceneElementsList()
{

   qtree->clear();
   unsigned int count = 0;
   for (unsigned int e = 0; e < SCENE->getSceneElements().size(); ++e)
   {
      count++;
      QString name = QString::fromStdString(SCENE->getSceneElements().at(e)->getName()); 
      if(SCENE->getSceneElements().at(e)->isStatic())
         name = name + QString(" [static]");
      QTreeWidgetItem* topLevel = new QTreeWidgetItem(QStringList(name));

      if(!SCENE->getSceneElements().at(e)->isStatic())
         for(unsigned int i = 0; i < SCENE->getSceneElements().at(e)->getNumInstances(); i++)
         {
            count++;
            QTreeWidgetItem* child = new QTreeWidgetItem(QStringList(QString("instance ") + QString::number(i)));
            topLevel->addChild(child);
            if(SCENE->getSceneElements().at(e)->isStatic())
            {
               child->setDisabled(true);
            }
         }

      qtree->addTopLevelItem(topLevel);
      if(SCENE->getSceneElements().at(e)->isStatic())
      {
         topLevel->setDisabled(true);
      }
      else
      {
         topLevel->setFlags(Qt::ItemIsEnabled);
         topLevel->setExpanded(true);
      }

   }

   qtree->setMaximumHeight(min(6, count)*16);

}

void PageScene::addCameraControl(const vector<Pose>& poses, int currentPoseIndex)
{
   QGroupBox* gbCamera = new QGroupBox("Camera");

   // Field of view

   QDoubleSpinBox* spinFovH = new QDoubleSpinBox();
   spinFovH->setRange(20.0, 100.0);
   spinFovH->setToolTip("Set camera's horizontal field of view.");
   spinFovH->setDecimals(1);
   spinFovH->setSingleStep(1.0);
   spinFovH->setValue(Scene::Instance()->getCamera()->getFovh());
   connect(spinFovH , SIGNAL(valueChanged(double)), SETTINGS, SLOT(setCameraFovH(double)));

   QDoubleSpinBox* spinZNear = new QDoubleSpinBox();
   spinZNear->setRange(0.00001, 10.0);
   spinZNear->setToolTip("Set camera's near plane.");
   spinZNear->setDecimals(5);
   spinZNear->setSingleStep(0.001);
   spinZNear->setValue(Scene::Instance()->getCamera()->getFrustum().zNear);
   connect(spinZNear , SIGNAL(valueChanged(double)), this, SLOT(setCameraZNear(double)));


   QHBoxLayout* hboxFov = new QHBoxLayout();
   hboxFov->addWidget(new QLabel("FovH"));
   hboxFov->addWidget(spinFovH);
   hboxFov->addWidget(new QLabel("ZNear"));
   hboxFov->addWidget(spinZNear);
   hboxFov->addStretch(1);

   gbCamera->setLayout(hboxFov);
   layout->addWidget(gbCamera);


}

void PageScene::addLightControl()
{
   QSlider* envSlider = new QSlider(Qt::Horizontal);
   envSlider->setRange(0, 359);
   envSlider->setToolTip("Environment map rotation angle in degrees");
   envSlider->setSingleStep(1);
   envSlider->setPageStep(1);
   envSlider->setTickPosition(QSlider::NoTicks);
   QLabel* envLabel = new QLabel("0");
   connect(envSlider, SIGNAL(valueChanged(int)), envLabel, SLOT(setNum(int)));
   connect(envSlider, SIGNAL(valueChanged(int)), this, SLOT(setEnvMapRotation(int)));
   envSlider->setValue(SCENE->getEnvMap()->getRotationAngle() / 3.14159f * 180);

   QHBoxLayout* hEnv = new QHBoxLayout();
   hEnv->addWidget(new QLabel("Rotation Angle"));
   hEnv->addWidget(envSlider);
   hEnv->addWidget(envLabel);

   QGroupBox* gbEnv = new QGroupBox("Environment Map");
   gbEnv->setLayout(hEnv);
   layout->addWidget(gbEnv);

   if(Scene::Instance()->getSpotLights().empty()) return;

   QGroupBox* gbLight = new QGroupBox("Spot Light(s)");
   QVBoxLayout* vbox = new QVBoxLayout();
  
   QHBoxLayout* hboxLight = new QHBoxLayout();

   QPushButton* pbPickLightColor = new QPushButton("Color");
   pbPickLightColor->setMinimumHeight(20);
   pbPickLightColor->setMaximumHeight(20);
   pbPickLightColor->setStyleSheet("* { background-color: rgb(255,255,230) }");
   pbPickLightColor->setToolTip("Set current spot light color and brightness");

   QPushButton* pbAddSpot    = new QPushButton("+");
   QPushButton* pbDeleteSpot = new QPushButton("-");
   pbAddSpot->setToolTip("Add a spot light. ['Spot light' must be activated in Mouse control]");
   pbDeleteSpot->setToolTip("Delete current spot light.");
   pbAddSpot->setMaximumHeight(20);
   pbDeleteSpot->setMaximumHeight(20);
   pbAddSpot->setMaximumWidth(30);
   pbDeleteSpot->setMaximumWidth(30);
   pbAddSpot->setStyleSheet("* { background-color: rgb(255,255,255) }");
   pbDeleteSpot->setStyleSheet("* { background-color: rgb(255,255,255) }");

   connect(pbAddSpot,    SIGNAL(clicked()), SETTINGS, SLOT(addSpotLight()));
   connect(pbDeleteSpot, SIGNAL(clicked()), SETTINGS, SLOT(deleteCurrentSpotLight()));

   hboxLight->addWidget(pbAddSpot);
   hboxLight->addWidget(pbDeleteSpot);

   connect(pbPickLightColor,    SIGNAL(clicked()), this, SLOT(setLightColor()));

   hboxLight->addWidget(pbPickLightColor);
   hboxLight->addStretch(1);
   connect(this, SIGNAL(lightColorChangeRequested(float, float, float, float)), SETTINGS, SLOT(forwardLightColorChangeRequest(float, float, float, float)));

   vbox->addLayout(hboxLight);

   QCheckBox* cbShadowMapping = new QCheckBox("Shadow Mapping");
   connect(cbShadowMapping, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleShadowMapping(bool)));
   cbShadowMapping->setChecked(true);

   vbox->addWidget(cbShadowMapping);

   QDoubleSpinBox* spinShadowEps = new QDoubleSpinBox();
   spinShadowEps ->setRange(0.0000, 10.0);
   spinShadowEps ->setToolTip("Set distance epsilon for shadow mapping");
   spinShadowEps ->setDecimals(5);
   spinShadowEps ->setSingleStep(0.00001);
   spinShadowEps ->setValue(SETTINGS->getShadowEpsilon());
   connect(spinShadowEps , SIGNAL(valueChanged(double)), SETTINGS, SLOT(setShadowEpsilon(double)));
   QHBoxLayout* hShadowEps = new QHBoxLayout();
   hShadowEps->addWidget(new QLabel("Shadow Epsilon"));
   hShadowEps->addWidget(spinShadowEps);
   hShadowEps->addStretch(1);

   if(Scene::Instance()->getNumSpotLights() > 0)
	   spinShadowEps->setValue(0.0001); 

   vbox->addLayout(hShadowEps);

   gbLight->setLayout(vbox);
   layout->addWidget(gbLight);

}

void PageScene::addModelControl()
{
   QGroupBox* gbModel = new QGroupBox("Scene Object(s)");
   QVBoxLayout* vbox = new QVBoxLayout();

   QCheckBox* cbHighlight = new QCheckBox("Highlight active model instance");
   connect(cbHighlight,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleHighlightModel(bool)));
   
   qtree = new QTreeWidget(this);
   qtree->setColumnCount(1);
   qtree->setHeaderHidden(true);
   connect(qtree, SIGNAL(currentItemChanged ( QTreeWidgetItem*, QTreeWidgetItem* )), this, SLOT(handleTreeItem(QTreeWidgetItem*, QTreeWidgetItem* )));

   // add instances for active dynamic element
   QPushButton* pbAddInstance = new QPushButton("Add Model Instance");
   pbAddInstance->setMaximumHeight(20);
   pbAddInstance->setStyleSheet("* { background-color: rgb(255,255,255) }");
   connect(pbAddInstance, SIGNAL(clicked()), SETTINGS, SLOT(forwardInstanceAddingRequest()));

   QPushButton* pbAnimation = new QPushButton("Animation");
   pbAnimation->setStyleSheet("* { background-color: rgb(230,255,210) }");
   pbAnimation->setMaximumHeight(20);
   pbAnimation->setCheckable(true);
   pbAnimation->setChecked(true);
   connect(pbAnimation, SIGNAL(clicked()), SETTINGS, SLOT(toggleAllObjAnimations()));

   QGroupBox* gbAutoRotate = new QGroupBox("Rotate active instance");
   gbAutoRotate->setCheckable(true);
   gbAutoRotate->setChecked(SETTINGS->autoRotateModel());
   connect(gbAutoRotate ,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAutoRotateModel(bool)));

   QRadioButton* rbXAxis = new QRadioButton("X axis");
   QRadioButton* rbYAxis = new QRadioButton("Y axis");
   QRadioButton* rbZAxis = new QRadioButton("Z axis");

   connect(rbXAxis ,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAutoRotateXAxis(bool)));
   connect(rbYAxis ,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAutoRotateYAxis(bool)));
   connect(rbZAxis ,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAutoRotateZAxis(bool)));

   QButtonGroup* bgAxis = new QButtonGroup();
   bgAxis->addButton(rbXAxis);
   bgAxis->addButton(rbYAxis);
   bgAxis->addButton(rbZAxis);

   QHBoxLayout* hAxis = new QHBoxLayout();
   hAxis->addWidget(rbXAxis);
   hAxis->addWidget(rbYAxis);
   hAxis->addWidget(rbZAxis);
   gbAutoRotate->setLayout(hAxis);

   rbYAxis->setChecked(true);


   vbox->addWidget(qtree);
   vbox->addWidget(cbHighlight);
   vbox->addWidget(gbAutoRotate);
   QHBoxLayout* hboxButtons = new QHBoxLayout();
   hboxButtons->addWidget(pbAnimation);
   hboxButtons->addSpacing(1);
   hboxButtons->addWidget(pbAddInstance);
   vbox->addLayout(hboxButtons);
   vbox->addStretch(1);

   gbModel->setLayout(vbox);
   layout->addWidget(gbModel);
}

void PageScene::addMouseControl()
{
   QGroupBox* gb = new QGroupBox("Mouse Control");
   QHBoxLayout* vbox = new QHBoxLayout();

   QRadioButton* rbCamera = new QRadioButton("Camera");
   connect(rbCamera, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleCameraActive(bool)));
   
   vbox->addWidget(rbCamera);

   if(Scene::Instance()->hasDynamicElements())
   {
      QRadioButton* rbModel = new QRadioButton("Dynamic\nObject");
      connect(rbModel,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleModelActive(bool)));
      vbox->addWidget(rbModel);
   }

   if(!Scene::Instance()->getSpotLights().empty())
   {
      QRadioButton* rbSpotLight  = new QRadioButton("Spot\nLight");
      connect(rbSpotLight,  SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleSpotLightActive(bool)));
      vbox->addWidget(rbSpotLight);
   }

   gb->setLayout(vbox);
   layout->addWidget(gb);

   rbCamera->setChecked(true);
}

void PageScene::addGBufferControl()
{
   // Misc
   QComboBox* gBuffer = new QComboBox();
   gBuffer->setStyleSheet("* { background-color: rgb(255,255,255) }");

   gBuffer->addItem("Positions");
   gBuffer->addItem("Normals");
   gBuffer->addItem("Material");
   gBuffer->addItem("Direct Light");
   gBuffer->addItem("(Combined)");

   gBuffer->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding)); 

   gBuffer->setCurrentIndex(SETTINGS->getCurrentGBufferTex());
   connect(gBuffer, SIGNAL(activated(int)), SETTINGS, SLOT(setCurrentGBufferTex(int)));

   // Put all GUI elements in one layout
   QHBoxLayout* hbox = new QHBoxLayout();
   hbox->addWidget(new QLabel("G-Buffer"));
   hbox->addWidget(gBuffer);

   layout->addLayout(hbox);
   layout->addStretch(3);

   QCheckBox* cbRender = new QCheckBox("Update Image");
   connect(cbRender, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleRenderingEnabled(bool)));
   cbRender->setChecked(true);
   layout->addWidget(cbRender);

}


// SLOTS

void PageScene::addCurrentPose()
{
   emit(poseAddingRequested());
   QString s("NEW POSE ");
   int newIndex = comboPoses->count();
   s += QString::number(newIndex);
   comboPoses->addItem(s);
   comboPoses->setCurrentIndex(newIndex);
}

void PageScene::deleteCurrentPose()
{
   if(comboPoses->count() > 1)
   {
      emit(poseDeletionRequested());

      // delete corresponding item in combobox
      int newIndex;
      if(comboPoses->currentIndex() == comboPoses->count()-1) // last
      {
         newIndex = comboPoses->currentIndex()-1;
      }
      else
      {
         newIndex = comboPoses->currentIndex();
      }
      comboPoses->removeItem(comboPoses->currentIndex());


   }
}

void PageScene::setLightColor()
{
   glm::vec3 currI;
   float currScale;
   //if(SETTINGS->spotLightActive())
   {
      currI = Scene::Instance()->getSpotLights().at(Scene::Instance()->getActiveSpotLightIndex())->getI();
   }
   /*else
   {
      return;
   }*/

   currScale = max(max(currI.x, currI.y), currI.z);
   if(currScale <= 1.0)
      currScale = 1.0;

   glm::ivec3 iCurrI = glm::ivec3(currI / currScale * 255.0f);

   QColor currColor(iCurrI.x, iCurrI.y, iCurrI.z);
   
   QColor c = QColorDialog::getColor(currColor);
   if(c.isValid())
   {
      bool ok;
      double scaleFactor = QInputDialog::getDouble(this, tr("Set light scale factor"),
         tr("Scale factor:"), currScale, 0.01, 1000, 2, &ok);
      if (ok)
      {
         cout << "color: " << c.redF() << " "<< c.greenF()<< " " << c.blueF() << endl;
         emit(lightColorChangeRequested(c.redF(), c.greenF(), c.blueF(), scaleFactor));
      }
   }
}

void PageScene::setCameraZNear(double zNear)
{
   Scene::Instance()->getCamera()->setZNear(float(zNear));
}

void PageScene::setEnvMapRotation(int angle)
{
   SCENE->getEnvMap()->setRotationAngle(angle);
}

void PageScene::handleTreeItem ( QTreeWidgetItem * current, QTreeWidgetItem * previous )
{
   if(!current || !previous) return;
 /*  if(current) std::cout << "current: " << current->text(0).toStdString() << std::endl;
   if(previous) std::cout << "prev: " << previous->text(0).toStdString() << std::endl;*/

   if(current->parent() == 0)
   {
      //std::cout << "Top Level Item" << std::endl;
      qtree->setCurrentItem(current->child(0));
   }
   else
   {
      // child
      int element = qtree->indexOfTopLevelItem(current->parent());
      int instance = current->parent()->indexOfChild(current);
      //std::cout << "element: " << element << std::endl;
      //std::cout << "instance: " << instance << std::endl;

      SCENE->setActiveInstance(element, instance);
   }
}