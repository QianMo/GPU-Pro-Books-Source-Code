#include "PageIndirectLight.h"

#include "Common.h"
#include "Qt/Settings.h"

PageIndirectLight::PageIndirectLight(int windowWidth, int windowHeight, QWidget* parent) : QWidget(parent)
, windowWidth(windowWidth), windowHeight(windowHeight)
{
   layout = new QVBoxLayout();
   setLayout(layout);

   addDisplayControl();
   addParameterControl();

   layout->addStretch(1);

}

void PageIndirectLight::addDisplayControl()
{
   //----------------------- Method Control --------------------/
   
   rbVGI = new QRadioButton("VGI Single Bounce");
   connect(rbVGI, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleVGIEnabled(bool)));

   rbAmbient = new QRadioButton("Ambient Term");
   connect(rbAmbient, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAmbientTermEnabled(bool)));

   QButtonGroup* bgMethod = new QButtonGroup();
   bgMethod->addButton(rbVGI);
   bgMethod->addButton(rbAmbient);
   rbVGI->setChecked(true);

   // ------------------------------- Group Box ----------------------------- /

   gbIndirectLight = new QGroupBox("Indirect Light");
   connect(gbIndirectLight, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleIndirectLightEnabled(bool)));
   connect(gbIndirectLight, SIGNAL(toggled(bool)), this, SLOT(toggleIndirectLightCheckBox(bool)));

   QVBoxLayout* vbox = new QVBoxLayout();
   vbox->addWidget(rbVGI);
   vbox->addWidget(rbAmbient);

   gbIndirectLight->setCheckable(true);
   gbIndirectLight->setChecked(false);
   gbIndirectLight->setChecked(true);
   gbIndirectLight->setLayout(vbox);
   layout->addWidget(gbIndirectLight);

   // ------------------------------- Parameters ----------------------------- /

   spinPatternSize = new QSpinBox();
   spinPatternSize->setToolTip("Size of the tiled random texture (square) for random ray rotation per pixel");
   spinPatternSize->setRange(2, 64);
   spinPatternSize->setSingleStep(2);
   spinPatternSize->setValue(SETTINGS->getRandomPatternSize()); // assume square size
   connect(spinPatternSize, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setRandomPatternSize(int)));
   connect(SETTINGS, SIGNAL(randomTextureSizeChanged(int)), spinPatternSize, SLOT(setValue(int)));

   QRadioButton* rbIL_E_ind = new QRadioButton("Indirect Illuminance (E_ind)");
   connect(rbIL_E_ind, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleIndirectLight_E_ind_Enabled(bool)));
   connect(SETTINGS, SIGNAL(toggledIndirectLight_E_ind(bool)), rbIL_E_ind, SLOT(setChecked(bool)));
   QRadioButton* rbIL_L_ind = new QRadioButton("Indirect Luminance (L_ind)");
   connect(rbIL_L_ind , SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleIndirectLight_L_ind_Enabled(bool)));
   connect(SETTINGS, SIGNAL(toggledIndirectLight_L_ind(bool)), rbIL_L_ind, SLOT(setChecked(bool)));
   QRadioButton* rbILCombine = new QRadioButton("Direct + Indirect Luminance");
   connect(rbILCombine, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleIndirectLightCombinationEnabled(bool)));
   connect(SETTINGS, SIGNAL(toggledIndirectLightCombination(bool)), rbILCombine, SLOT(setChecked(bool)));


   QHBoxLayout* hLdir = new QHBoxLayout();
   hLdir->addWidget(rbILCombine);
   QVBoxLayout* vbox2 = new QVBoxLayout();
   vbox2->addWidget(rbIL_E_ind);
   vbox2->addWidget(rbIL_L_ind);
   vbox2->addLayout(hLdir);

   rbIL_E_ind->setChecked(SETTINGS->indirectLight_E_ind_Enabled());
   rbIL_L_ind->setChecked(SETTINGS->indirectLight_L_ind_Enabled());
   rbILCombine->setChecked(SETTINGS->indirectLightCombinationEnabled());

   QGroupBox* gbResult = new QGroupBox("Result");
   gbResult->setLayout(vbox2);

   layout->addWidget(gbResult);

   comboBufferSize = new QComboBox();
   comboBufferSize->setStyleSheet("* { background-color: rgb(255,255,255) }");

   comboBufferSize->addItem("FULL");
   comboBufferSize->addItem("HALF");
   comboBufferSize->addItem("QUARTER");
   comboBufferSize->setToolTip("Resolution of indirect light buffer");
   connect(comboBufferSize, SIGNAL(currentIndexChanged(int)), SETTINGS, SLOT(setCurrentILBufferSize(int)));
   connect(SETTINGS, SIGNAL(changedILBufferSize(int)), comboBufferSize, SLOT(setCurrentIndex(int)));
   comboBufferSize->setCurrentIndex(SETTINGS->getCurrentILBufferSize());

   QHBoxLayout* hBuff = new QHBoxLayout();
   hBuff->addWidget(new QLabel("Buffer Size"));
   hBuff->addWidget(comboBufferSize);
   hBuff->addStretch(1);

   layout->addLayout(hBuff);

}

void PageIndirectLight::addParameterControl()
{
   layout->addSpacing(20);
   layout->addWidget(new QLabel("<b>Parameters:</b>"));

	raysSlider = new QSlider(Qt::Horizontal);
	raysSlider->setRange(1, MAX_RAYS);
   raysSlider->setToolTip("Number of rays per pixel");
	raysSlider->setSingleStep(1);
	raysSlider->setPageStep(1);
   raysSlider->setTickPosition(QSlider::NoTicks);
	QLabel* raysLabel = new QLabel("0");
	connect(raysSlider, SIGNAL(valueChanged(int)), raysLabel, SLOT(setNum(int)));
	connect(raysSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setNumRays(int)));
	connect(SETTINGS, SIGNAL(numRaysChanged(int)), raysSlider, SLOT(setValue(int)));
   raysSlider->setValue(SETTINGS->getNumRays());

	stepsSlider = new QSlider(Qt::Horizontal);
   stepsSlider->setToolTip("Maximum number of voxel hierarchy traversal iterations");
	stepsSlider->setRange(1, 256);
	stepsSlider->setSingleStep(1);
	stepsSlider->setPageStep(1);
   stepsSlider->setTickPosition(QSlider::NoTicks);
	QLabel* stepsLabel = new QLabel("0");
	connect(stepsSlider, SIGNAL(valueChanged(int)), stepsLabel, SLOT(setNum(int)));
	connect(stepsSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setNumSteps(int)));
	connect(SETTINGS, SIGNAL(numStepsChanged(int)), stepsSlider, SLOT(setValue(int)));
   stepsSlider->setValue(SETTINGS->getNumSteps());

	resolutionSlider = new QSlider(Qt::Horizontal);
	resolutionSlider->setRange(4, 10);
   resolutionSlider->setToolTip("Resolution of the voxel texture in x and y-direction");
	resolutionSlider->setSingleStep(1);
	resolutionSlider->setPageStep(1);
   resolutionSlider->setTickPosition(QSlider::NoTicks);
	resolutionLabel = new QLabel("0");
   connect(resolutionSlider, SIGNAL(valueChanged(int)), this, SLOT(setVoxelResolution(int)));
   connect(this, SIGNAL(changedVoxelResolution(int)), SETTINGS, SLOT(setVoxelTextureResolution(int)));
   connect(SETTINGS, SIGNAL(voxelTextureResolutionChanged(int)), this, SLOT(setVoxelResolutionSlider(int)));
   resolutionSlider->setValue(7);

   spinRadius = new QDoubleSpinBox();
   spinRadius->setRange(0.05, 100.0);
   spinRadius->setToolTip("Set maximum length of rays.");
   spinRadius->setDecimals(2);
   spinRadius->setSingleStep(0.05);
   spinRadius->setValue(SETTINGS->getRadius());
   connect(spinRadius, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setRadius(double)));
	connect(SETTINGS, SIGNAL(radiusChanged(double)), spinRadius, SLOT(setValue(double)));

   QDoubleSpinBox* spinSpread = new QDoubleSpinBox();
   spinSpread->setRange(0.01, 1.0);
   spinSpread->setToolTip("Set spread (1.0 = full hemisphere).");
   spinSpread->setDecimals(2);
   spinSpread->setSingleStep(0.01);
   spinSpread->setValue(SETTINGS->getSpread());
   connect(spinSpread, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setSpread(double)));
	connect(SETTINGS, SIGNAL(spreadChanged(double)), spinSpread, SLOT(setValue(double)));

   spinLdirScale = new QDoubleSpinBox();
   spinLdirScale->setRange(0.01, 10.0);
   spinLdirScale->setToolTip("Set scale factor for direct light");
   spinLdirScale->setDecimals(2);
   spinLdirScale->setSingleStep(0.05);
   spinLdirScale->setValue(SETTINGS->getDirectLightScaleFactor());
   connect(spinLdirScale, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setDirectLightScaleFactor(double)));
   connect(SETTINGS, SIGNAL(directLightScaleFactorChanged(double)), spinLdirScale, SLOT(setValue(double)));

   spinLindirScale = new QDoubleSpinBox();
   spinLindirScale->setRange(0.0, 100.0);
   spinLindirScale->setToolTip("Set scale factor for indirect light");
   spinLindirScale->setDecimals(1);
   spinLindirScale->setSingleStep(0.1);
   spinLindirScale->setValue(SETTINGS->getIndirectLightScaleFactor());
   connect(spinLindirScale, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setIndirectLightScaleFactor(double)));
   connect(SETTINGS, SIGNAL(indirectLightScaleFactorChanged(double)), spinLindirScale, SLOT(setValue(double)));

   spinEnvMapBrightness = new QDoubleSpinBox();
   spinEnvMapBrightness->setRange(0.0, 100.0);
   spinEnvMapBrightness->setDecimals(4);
   spinEnvMapBrightness->setToolTip("Scales the brightness of the environment map.");
   spinEnvMapBrightness->setSingleStep(0.1);
   spinEnvMapBrightness->setValue(SETTINGS->getEnvMapBrightness());
   connect(spinEnvMapBrightness, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setEnvMapBrightness(double)));
   connect(SETTINGS, SIGNAL(envMapBrightnessChanged(double)), spinEnvMapBrightness, SLOT(setValue(double)));

   spinOcclusionStrength = new QDoubleSpinBox();
   spinOcclusionStrength->setRange(0.0, 100.0);
   spinOcclusionStrength->setToolTip("Higher values than 1.0 will darken occluded areas");
   spinOcclusionStrength->setDecimals(4);
   spinOcclusionStrength->setSingleStep(0.01);
   spinOcclusionStrength->setValue(SETTINGS->getOcclusionStrength());
   connect(spinOcclusionStrength, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setOcclusionStrength(double)));
   connect(SETTINGS, SIGNAL(occlusionStrengthChanged(double)), spinOcclusionStrength, SLOT(setValue(double)));

   QDoubleSpinBox* spinDistanceEps = new QDoubleSpinBox();
   spinDistanceEps->setRange(0.01, 30.0);
   spinDistanceEps->setToolTip("Map lookup (RSM): Set threshold scale for valid light samples.");
   spinDistanceEps->setDecimals(2);
   spinDistanceEps->setSingleStep(0.1);
   spinDistanceEps->setValue(SETTINGS->getDistanceThresholdScale());
   connect(spinDistanceEps, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setDistanceThresholdScale(double)));
   connect(SETTINGS, SIGNAL(distanceThresholdScaleChanged(double)), spinDistanceEps, SLOT(setValue(double)));

   QDoubleSpinBox* spinCosThetaOffsetScale = new QDoubleSpinBox();
   spinCosThetaOffsetScale->setRange(0.01, 10.0);
   spinCosThetaOffsetScale->setToolTip("Set scale factor for offsetting the ray starting point according to the angle between the ray and the normal.");
   spinCosThetaOffsetScale->setDecimals(2);
   spinCosThetaOffsetScale->setSingleStep(0.05);
   spinCosThetaOffsetScale->setValue(SETTINGS->getVoxelOffsetCosThetaScale());
   connect(spinCosThetaOffsetScale, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setVoxelOffsetCosThetaScale(double)));
   connect(SETTINGS, SIGNAL(voxelOffsetCosThetaScaleChanged(double)), spinCosThetaOffsetScale, SLOT(setValue(double)));
 
   
   QDoubleSpinBox* spinNormalOffsetScale = new QDoubleSpinBox();
   spinNormalOffsetScale->setRange(0.01, 10.0);
   spinNormalOffsetScale->setToolTip("Set scale factor for offsetting the ray starting point along the the normal.");
   spinNormalOffsetScale->setDecimals(2);
   spinNormalOffsetScale->setSingleStep(0.05);
   spinNormalOffsetScale->setValue(SETTINGS->getVoxelOffsetNormalScale());
   connect(spinNormalOffsetScale, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setVoxelOffsetNormalScale(double)));
   connect(SETTINGS, SIGNAL(voxelOffsetNormalScaleChanged(double)), spinNormalOffsetScale, SLOT(setValue(double)));
   
   QGridLayout* gridContrast = new QGridLayout();
   gridContrast->addWidget(new QLabel("Indirect Light Scale"), 0, 0);
   gridContrast->addWidget(spinLindirScale, 0, 1);
   gridContrast->addWidget(new QLabel("Direct Light Scale"), 1, 0);
   gridContrast->addWidget(spinLdirScale, 1, 1);
   gridContrast->addWidget(new QLabel("Env Map Occlusion Strength"), 2, 0);
   gridContrast->addWidget(spinOcclusionStrength, 2, 1);
   gridContrast->addWidget(new QLabel("Env Map Brightness"), 3, 0);
   gridContrast->addWidget(spinEnvMapBrightness, 3, 1);


   QGridLayout* grid = new QGridLayout();
   grid->addWidget(new QLabel("Rays"), 0, 0);
   grid->addWidget(raysSlider, 0, 1);
   grid->addWidget(raysLabel, 0, 2);
   grid->addWidget(new QLabel("Steps"), 1, 0);
   grid->addWidget(stepsSlider, 1, 1);
   grid->addWidget(stepsLabel, 1, 2);
   grid->addWidget(new QLabel("Voxel Resolution"), 2, 0);
   grid->addWidget(resolutionSlider, 2, 1);
   grid->addWidget(resolutionLabel, 2, 2);

   grid->addWidget(new QLabel("Radius"), 3, 0);
   grid->addWidget(spinRadius, 3, 1);
   grid->addWidget(new QLabel("Spread"), 4, 0);
   grid->addWidget(spinSpread, 4, 1);

   QGridLayout* gridOffsets = new QGridLayout();
   gridOffsets->addWidget(new QLabel("RayOffsetScale"), 0, 0);
   gridOffsets->addWidget(spinCosThetaOffsetScale, 0, 1);
   gridOffsets->addWidget(new QLabel("NormalOffsetScale"), 1, 0);
   gridOffsets->addWidget(spinNormalOffsetScale, 1, 1);
   gridOffsets->addWidget(new QLabel("Distance Threshold\n (Map Lookup)"), 2, 0);
   gridOffsets->addWidget(spinDistanceEps, 2, 1);

   
   QTabWidget* tabWidget = new QTabWidget();
   //1. Create a QTabWidget.
   //2. Create a QWidget for each of the pages in the tab dialog, but do not specify parent widgets for them.
   //3. Insert child widgets into the page widget, using layouts to position them as normal.
   //4. Call addTab() or insertTab() to put the page widgets into the tab widget, giving each tab a suitable label with an optional keyboard shortcut.
   
   QHBoxLayout* hRandTex = new QHBoxLayout();
   hRandTex->addWidget(new QLabel("Random Texture\n Size"));
   hRandTex->addWidget(spinPatternSize);
   
   QVBoxLayout* vGeneral = new QVBoxLayout();
   vGeneral->addLayout(grid);
   vGeneral->addSpacing(10);
   vGeneral->addLayout(hRandTex);

   QVBoxLayout* vOffsets = new QVBoxLayout();
   vOffsets->addLayout(gridOffsets);
   vOffsets->addStretch(1);

   QVBoxLayout* vContrast = new QVBoxLayout();
   vContrast->addLayout(gridContrast);
   vContrast->addStretch(1);


   QWidget* tabGeneral = new QWidget();
   tabGeneral->setLayout(vGeneral);
   QWidget* tabOffsets = new QWidget();
   tabOffsets->setLayout(vOffsets);
   QWidget* tabContrast = new QWidget();
   tabContrast->setLayout(vContrast);
   tabWidget->addTab(tabGeneral, "General");
   tabWidget->addTab(tabContrast, "Brightness");
   tabWidget->addTab(tabOffsets, "Offsets");
   tabWidget->setTabPosition(QTabWidget::North);

   tabWidget->setPalette(QPalette(QColor(245, 245, 246)));


   layout->addWidget(tabWidget);


}

void PageIndirectLight::setVoxelResolution(int level)
{
   int resolution = pow(2.0, level);
   resolutionLabel->setNum(resolution);
   
   emit(changedVoxelResolution(resolution));
}

void PageIndirectLight::setVoxelResolutionSlider(int resolution)
{
   int level = log(double(resolution)) / log(2.0);
   resolutionSlider->setValue(level);
}

void PageIndirectLight::toggleIndirectLightCheckBox(bool checked)
{
   if(gbIndirectLight->isChecked() != checked)
   {
      gbIndirectLight->setChecked(checked);
   }
   emit(toggledIndirectLightCheckBox(checked));
}
