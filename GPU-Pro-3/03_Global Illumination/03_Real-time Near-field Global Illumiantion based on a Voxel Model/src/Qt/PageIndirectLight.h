#ifndef PAGEINDIRECTLIGHT_H
#define PAGEINDIRECTLIGHT_H

#include <QtGui>

/// A page widget that will be added to the stacked widget in the central widget.
/// It controls the settings for indirect light computation.
class PageIndirectLight : public QWidget
{
   Q_OBJECT

public:
   PageIndirectLight(int windowWidth, int windowHeight, QWidget* parent = 0);

   void addDisplayControl();
   void addParameterControl();

public slots:
   void toggleIndirectLightCheckBox(bool checked);
   void setVoxelResolution(int level);
   void setVoxelResolutionSlider(int resolution);

signals:
   void forwardedPatternSize(int size);
   void toggledIndirectLightCheckBox(bool checked);
   void changedVoxelResolution(int res);

private:
   QSpinBox* spinPatternSize;


   // RadioButtons for choosing Indirect Light Method
   QGroupBox* gbIndirectLight;

   QRadioButton* rbVGI;

   QRadioButton* rbAmbient;

   //QCheckBox* cb2DRand;

   // Parameters
   QComboBox* comboBufferSize;
   QSlider* raysSlider;
   QSlider* stepsSlider;
   QSlider* resolutionSlider;
   QLabel* resolutionLabel;
   QDoubleSpinBox* spinRadius;
   QDoubleSpinBox* spinLindirScale;
   QDoubleSpinBox* spinLdirScale;

   QDoubleSpinBox* spinOcclusionStrength;
   QDoubleSpinBox* spinEnvMapBrightness;

   QVBoxLayout* layout;

   int windowWidth, windowHeight;


};

#endif
