#ifndef PAGEDEBUGGING_H
#define PAGEDEBUGGING_H

#include <QtGui>

/// A page widget that will be added to the stacked widget in the central widget.
/// It controls debugging graphics output.
class PageDebugging : public QWidget
{
   Q_OBJECT

public:
   PageDebugging(QWidget* parent = 0);

public slots:
   void toggleMipmapTestVisEnabled(bool enabled);

signals:
   void toggledMipmapTestVisEnabled(bool enabled);

private:

   void addDebuggingGraphicsControl();
   QGridLayout* createStartEndPointControl();

   QVBoxLayout* layout;

   QLabel* cycleIndexLabel;
   QLabel* cycleSliderLabel;
   QSlider* cycleSlider;
   QCheckBox* cbShowVoxelCubes;
   QCheckBox* cbMipmapTestVis;
};

#endif
