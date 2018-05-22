#include "PageFilter.h"

#include "Common.h"
#include "Qt/Settings.h"

PageFilter::PageFilter(QWidget* parent) : QWidget(parent)
{
   layout = new QVBoxLayout();
   setLayout(layout);

   addFilterControl();

   layout->addStretch(2);

}

void PageFilter::addFilterControl()
{
   cbFilterOn = new QCheckBox("Use Filter (geometry-sensitive)");
   connect(cbFilterOn, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleFilterEnabled(bool)));
   connect(SETTINGS, SIGNAL(filterToggled(bool)), cbFilterOn, SLOT(setChecked(bool)));
   cbFilterOn->setChecked(SETTINGS->filterEnabled());

   QFrame* line = new QFrame();
   line->setFrameShape(QFrame::HLine);
   line->setFrameShadow(QFrame::Sunken);

   layout->addWidget(cbFilterOn);

	radiusSlider = new QSlider(Qt::Horizontal);
	radiusSlider->setRange(1, 32);
	radiusSlider->setSingleStep(1);
	radiusSlider->setPageStep(1);
   radiusSlider->setTickPosition(QSlider::NoTicks);
	QLabel* radiusLabel = new QLabel("0");
	connect(radiusSlider, SIGNAL(valueChanged(int)), radiusLabel, SLOT(setNum(int)));
	connect(radiusSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setFilterRadius(int)));
   radiusSlider->setValue(SETTINGS->getFilterRadius());
   connect(SETTINGS, SIGNAL(filterRadiusChanged(int)), radiusSlider, SLOT(setValue(int)));

	iterRadiusSlider = new QSlider(Qt::Horizontal);
	iterRadiusSlider->setRange(1, 32);
	iterRadiusSlider->setSingleStep(1);
	iterRadiusSlider->setPageStep(1);
   iterRadiusSlider->setTickPosition(QSlider::NoTicks);
	QLabel* iterRadiusLabel = new QLabel("0");
	connect(iterRadiusSlider, SIGNAL(valueChanged(int)), iterRadiusLabel, SLOT(setNum(int)));
	connect(iterRadiusSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setFilterIterationRadius(int)));
   iterRadiusSlider->setValue(SETTINGS->getFilterIterationRadius());
   connect(SETTINGS, SIGNAL(filterIterationRadiusChanged(int)), iterRadiusSlider, SLOT(setValue(int)));


	iterSlider = new QSlider(Qt::Horizontal);
	iterSlider ->setRange(1, 10);
	iterSlider ->setSingleStep(1);
	iterSlider ->setPageStep(1);
   iterSlider ->setTickPosition(QSlider::NoTicks);
	QLabel* iterLabel = new QLabel("1");
	connect(iterSlider, SIGNAL(valueChanged(int)), iterLabel, SLOT(setNum(int)));
	connect(iterSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setFilterIterations(int)));
   iterSlider->setValue(SETTINGS->getFilterIterations());
   connect(SETTINGS, SIGNAL(filterIterationsChanged(int)), iterSlider, SLOT(setValue(int)));

   spinDistLimit = new QDoubleSpinBox();
   spinDistLimit->setRange(0.00, 5.0);
   spinDistLimit->setToolTip("Set distance limit (treshold).");
   spinDistLimit->setDecimals(2);
   spinDistLimit->setSingleStep(0.05);
   spinDistLimit->setValue(SETTINGS->getFilterDistanceLimit());
   connect(spinDistLimit , SIGNAL(valueChanged(double)), SETTINGS, SLOT(setFilterDistanceLimit(double)));
   connect(SETTINGS, SIGNAL(filterDistanceLimitChanged(double)), spinDistLimit, SLOT(setValue(double)));

   spinNormalLimit = new QDoubleSpinBox();
   spinNormalLimit->setRange(0, 90);
   spinNormalLimit->setToolTip("Set up to which angle between two normals they are considered as similar.");
   spinNormalLimit->setDecimals(2);
   spinNormalLimit->setSingleStep(0.5);
   connect(spinNormalLimit, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setFilterNormalLimit(double)));
   spinNormalLimit->setValue(10);
   connect(SETTINGS, SIGNAL(filterNormalLimitChanged(double)), spinNormalLimit, SLOT(setValue(double)));

   QDoubleSpinBox* spinMaterialLimit = new QDoubleSpinBox();
   spinMaterialLimit->setRange(0, 2);
   spinMaterialLimit->setToolTip("Maximal euclidean distance between two colors.");
   spinMaterialLimit->setDecimals(2);
   spinMaterialLimit->setSingleStep(0.01);
   connect(spinMaterialLimit, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setFilterMaterialLimit(double)));
   spinMaterialLimit->setValue(0.3);
   connect(SETTINGS, SIGNAL(filterMaterialLimitChanged(double)), spinMaterialLimit, SLOT(setValue(double)));

   QHBoxLayout* hbox1a = new QHBoxLayout();
   hbox1a->addWidget(new QLabel("Radius"));
   hbox1a->addWidget(radiusSlider);
   hbox1a->addWidget(radiusLabel);
   layout->addLayout(hbox1a);

   QHBoxLayout* hbox1aa = new QHBoxLayout();
   hbox1aa->addWidget(new QLabel("Iter. Radius"));
   hbox1aa->addWidget(iterRadiusSlider);
   hbox1aa->addWidget(iterRadiusLabel);
   layout->addLayout(hbox1aa);

   QHBoxLayout* hbox1b = new QHBoxLayout();
   hbox1b->addWidget(new QLabel("Iterations"));
   hbox1b->addWidget(iterSlider);
   hbox1b->addWidget(iterLabel);
   layout->addLayout(hbox1b);

   layout->addSpacing(10);

   QHBoxLayout* hbox2 = new QHBoxLayout();
   hbox2->addWidget(new QLabel("Distance Limit"));
   hbox2->addWidget(spinDistLimit);
   layout->addLayout(hbox2);

   QHBoxLayout* hbox3 = new QHBoxLayout();
   hbox3->addWidget(new QLabel("Normal Limit"));
   hbox3->addWidget(spinNormalLimit);
   layout->addLayout(hbox3);

   QHBoxLayout* hbox4 = new QHBoxLayout();
   hbox4->addWidget(new QLabel("Material Limit"));
   hbox4->addWidget(spinMaterialLimit);
   layout->addLayout(hbox4);

   QCheckBox* cbSurfaceDetail = new QCheckBox("Surface Detail");
   connect(cbSurfaceDetail, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleSurfaceDetailEnabled(bool)));
   connect(SETTINGS, SIGNAL(surfaceDetailToggled(bool)), cbSurfaceDetail, SLOT(setChecked(bool)));
   cbSurfaceDetail->setChecked(SETTINGS->surfaceDetailEnabled());
   
   QDoubleSpinBox* spinSurfaceDetailAlpha = new QDoubleSpinBox();
   spinSurfaceDetailAlpha->setRange(0, 1);
   spinSurfaceDetailAlpha->setDecimals(2);
   spinSurfaceDetailAlpha->setSingleStep(0.05);
   connect(spinSurfaceDetailAlpha, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setSurfaceDetailAlpha(double)));
   spinSurfaceDetailAlpha->setValue(SETTINGS->getSurfaceDetailAlpha());
   connect(SETTINGS, SIGNAL(surfaceDetailAlphaChanged(double)), spinSurfaceDetailAlpha, SLOT(setValue(double)));

   
   QHBoxLayout* hSurface = new QHBoxLayout();
   hSurface->addWidget(cbSurfaceDetail);
   hSurface->addWidget(spinSurfaceDetailAlpha);

   layout->addSpacing(10);
   layout->addLayout(hSurface);
   layout->addStretch(1);

   layout->addStretch(1);

}
