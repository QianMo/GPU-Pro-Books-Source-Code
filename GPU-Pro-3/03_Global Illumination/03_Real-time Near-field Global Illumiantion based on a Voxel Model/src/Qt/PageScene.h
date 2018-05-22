#ifndef PAGESCENE_H
#define PAGESCENE_H

#include <string>
#include <sstream>
#include <QtGui>

#include "Scene/SceneDataStructs.h"

/// A page widget that will be added to the stacked widget in the central widget.
/// It controls the settings for scene elements such as light, model, camera.
class PageScene : public QWidget
{
   Q_OBJECT

public:
   PageScene(const vector<Pose>& poses, int currentPoseIndex, QWidget* parent = 0);

public slots:
   void addCurrentPose();
   void deleteCurrentPose();
   void setLightColor();
   void addSceneElementsList();
   void setCameraZNear(double);
   void setEnvMapRotation(int);

   void handleTreeItem ( QTreeWidgetItem * current, QTreeWidgetItem * previous );

signals:
   void poseAddingRequested();
   void poseDeletionRequested();

   void lightColorChangeRequested(float r, float g, float b, float scaleFactor);

private:
   void addMouseControl();

   void addModelControl();
   void addCameraControl(const vector<Pose>& poses, int currentPoseIndex);
   void addLightControl();
   void addGBufferControl();



   QVBoxLayout* layout;
   QTreeWidget* qtree;

   QComboBox* comboPoses;


};

#endif
