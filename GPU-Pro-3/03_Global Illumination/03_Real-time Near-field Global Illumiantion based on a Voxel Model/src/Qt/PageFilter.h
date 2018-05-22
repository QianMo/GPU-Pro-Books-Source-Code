#ifndef PAGEFILTER_H
#define PAGEFILTER_H

#include <QtGui>

class PageFilter: public QWidget
{
   Q_OBJECT

public:
   PageFilter(QWidget* parent = 0);

signals:

private:

   void addFilterControl();

   QVBoxLayout* layout;

   // Parameter GUI Elements
   QCheckBox* cbFilterOn;
   QSlider* radiusSlider;
   QSlider* iterRadiusSlider;
   QSlider* iterSlider;
   QDoubleSpinBox* spinDistLimit;
   QDoubleSpinBox* spinNormalLimit;

};

#endif
