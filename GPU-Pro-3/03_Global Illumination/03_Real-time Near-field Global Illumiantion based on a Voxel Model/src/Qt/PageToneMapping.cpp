#include "PageToneMapping.h"

#include "Qt/Settings.h"
#include "Common.h"

PageToneMapping::PageToneMapping(QWidget* parent) : QWidget(parent)
{
   layout = new QVBoxLayout();
   setLayout(layout);

   addTonemapControl();
   addMiscControl();

   layout->addStretch(2);

}

void PageToneMapping::addTonemapControl()
{
   QGroupBox* gbTonemapOn = new QGroupBox("Tone Mapping");
   gbTonemapOn->setCheckable(true);
   connect(gbTonemapOn, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleToneMappingEnabled(bool)));
   connect(SETTINGS, SIGNAL(toneMappingToggled(bool)), gbTonemapOn, SLOT(setChecked(bool)));
   gbTonemapOn->setChecked(SETTINGS->toneMappingEnabled());
   
   QRadioButton* rbLinear = new QRadioButton("Linear");
   connect(rbLinear, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleLinearToneMappingEnabled(bool)));
   connect(SETTINGS, SIGNAL(linearToneMappingToggled(bool)), rbLinear, SLOT(setChecked(bool)));

   QRadioButton* rbLog = new QRadioButton("Log");
   connect(rbLog, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleLogToneMappingEnabled(bool)));
   connect(SETTINGS, SIGNAL(logToneMappingToggled(bool)), rbLog, SLOT(setChecked(bool)));

   QDoubleSpinBox* spinSimpleMaxRad = new QDoubleSpinBox();
   spinSimpleMaxRad ->setRange(0.05, 100000.0);
   spinSimpleMaxRad ->setDecimals(2);
   spinSimpleMaxRad ->setSingleStep(0.05);
   spinSimpleMaxRad ->setValue(SETTINGS->getSimpleMaxRadiance());
   connect(spinSimpleMaxRad , SIGNAL(valueChanged(double)), SETTINGS, SLOT(setSimpleMaxRadiance(double)));
   connect(SETTINGS, SIGNAL(simpleMaxRadianceChanged(double)), spinSimpleMaxRad, SLOT(setValue(double)));

   QHBoxLayout* hSimpleMax = new QHBoxLayout();
   hSimpleMax->addWidget(new QLabel("Max Radiance"));
   hSimpleMax->addWidget(spinSimpleMaxRad);

   QVBoxLayout* vbox = new QVBoxLayout();
   vbox->addWidget(rbLinear);
   vbox->addWidget(rbLog);
   vbox->addLayout(hSimpleMax);

   rbLog->setChecked(true);

   gbTonemapOn->setLayout(vbox);
   layout->addWidget(gbTonemapOn);

}

void PageToneMapping::addMiscControl()
{
   QDoubleSpinBox* spinGamma = new QDoubleSpinBox();
   spinGamma->setRange(1.0, 4.0);
   spinGamma->setToolTip("Set gamma exponent for gamma correction.");
   spinGamma->setDecimals(2);
   spinGamma->setSingleStep(0.05);
   spinGamma->setValue(SETTINGS->getGammaExponent());
   connect(spinGamma , SIGNAL(valueChanged(double)), SETTINGS, SLOT(setGammaExponent(double)));


   QHBoxLayout* hbox = new QHBoxLayout();
   hbox->addWidget(new QLabel("Gamma Exponent"));
   hbox->addWidget(spinGamma);

   layout->addLayout(hbox);

}