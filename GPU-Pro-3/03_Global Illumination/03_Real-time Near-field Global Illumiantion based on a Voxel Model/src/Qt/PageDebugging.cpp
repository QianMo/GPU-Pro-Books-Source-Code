#include "PageDebugging.h"

#include "Common.h"
#include "Qt/Settings.h"

PageDebugging::PageDebugging(QWidget* parent) : QWidget(parent)
{
   layout = new QVBoxLayout();
   setLayout(layout);

   addDebuggingGraphicsControl();

   layout->addStretch(1);

}

void PageDebugging::addDebuggingGraphicsControl()
{

   // Voxels
   QGroupBox* gbVisualizeVoxels = new QGroupBox("Enable Voxelization");
   gbVisualizeVoxels->setCheckable(true);
   gbVisualizeVoxels->setChecked(false);
   connect(gbVisualizeVoxels, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleVoxelizationEnabled(bool)));

   // Voxel Visualization Method
   QRadioButton* rbRayCasting = new QRadioButton("Ray Casting");
   connect(rbRayCasting, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleBinaryRayCasting(bool)));
   QRadioButton* rbCubes = new QRadioButton("Cubes");
   connect(rbCubes, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleVoxelCubes(bool)));
   QRadioButton* rbNone = new QRadioButton("None");
   rbNone->setToolTip("Voxelize in background, do not visualize voxels");
   rbCubes->setChecked(true);

   QHBoxLayout* hVisVoxels = new QHBoxLayout();
   hVisVoxels->addWidget(rbNone);
   hVisVoxels->addWidget(rbCubes);
   hVisVoxels->addWidget(rbRayCasting);

   QButtonGroup* bgVoxelVis = new QButtonGroup();
   bgVoxelVis->addButton(rbRayCasting);
   bgVoxelVis->addButton(rbCubes);
   bgVoxelVis->addButton(rbNone);

   // Voxelization Algorithm
   QRadioButton* rbAtlasBinary = new QRadioButton("Atlas (Binary)");
   connect(rbAtlasBinary, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleAtlasBinaryVoxelizationEnabled(bool)));
   rbAtlasBinary->setChecked(true);

   QCheckBox* cbMipmapping = new QCheckBox("Create Mipmaps");
   connect(cbMipmapping, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleMipmapping(bool)));
   cbMipmapping->setChecked(true);
   QCheckBox* cbPrevox = new QCheckBox("Pre-Voxelization");
   connect(cbPrevox, SIGNAL(toggled(bool)), SETTINGS, SLOT(togglePreVoxelization(bool)));
   cbPrevox->setChecked(true);

   QSlider* levelSlider = new QSlider(Qt::Horizontal);
   levelSlider->setRange(0, 16);
   levelSlider->setSingleStep(1);
   levelSlider->setPageStep(1);
   levelSlider->setTickPosition(QSlider::NoTicks);
	QLabel* levelLabel = new QLabel("0");
	connect(levelSlider, SIGNAL(valueChanged(int)), levelLabel, SLOT(setNum(int)));
	connect(levelSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setMipmapLevel(int)));
   levelSlider->setValue(0);

   cycleSlider = new QSlider(Qt::Horizontal);
   cycleSlider->setRange(0, 256);
   cycleSlider->setSingleStep(1);
   cycleSlider->setPageStep(1);
   cycleSlider->setTickPosition(QSlider::NoTicks);
	cycleIndexLabel = new QLabel("0");
	connect(cycleSlider, SIGNAL(valueChanged(int)), cycleIndexLabel, SLOT(setNum(int)));
	connect(cycleSlider, SIGNAL(valueChanged(int)), SETTINGS, SLOT(setMipmapTestCycleIndex(int)));
   cycleSlider->setValue(0);

   QHBoxLayout* hboxLevel = new QHBoxLayout();
   hboxLevel->addWidget(new QLabel("MipmapLevel"));
   hboxLevel->addWidget(levelSlider);
   hboxLevel->addWidget(levelLabel);

   // ---------------- Mipmap test --------------------------------

   cycleSliderLabel = new QLabel("Mipmap Test\nCycle Index");
   QHBoxLayout* hboxCycle = new QHBoxLayout();
   hboxCycle->addWidget(cycleSliderLabel);
   hboxCycle->addWidget(cycleSlider);
   hboxCycle->addWidget(cycleIndexLabel);

   cbShowVoxelCubes = new QCheckBox("Show Voxel Cubes");
   connect(cbShowVoxelCubes, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleShowVoxelCubesDuringMipmapTest(bool)));
   cbShowVoxelCubes->setChecked(true);
   
   cycleSliderLabel->setEnabled(false);
   cycleSlider->setEnabled(false);
   cycleIndexLabel->setEnabled(false);
   cbShowVoxelCubes->setEnabled(false);

   cbMipmapTestVis = new QCheckBox("Mipmap Intersection Test Visualization");
   connect(cbMipmapTestVis, SIGNAL(toggled(bool)), this, SLOT(toggleMipmapTestVisEnabled(bool)));
   connect(this, SIGNAL(toggledMipmapTestVisEnabled(bool)), SETTINGS, SLOT(toggleMipmapTestVisualization(bool)));
   cbMipmapTestVis->setChecked(false);

   QGridLayout* gbStartEnd = createStartEndPointControl();



   // ------------------------------------------------------------

   QFrame* separator0 = new QFrame( this );
   QFrame* separator2 = new QFrame( this );
   separator0 ->setFrameStyle( QFrame::HLine | QFrame::Sunken );
   separator2 ->setFrameStyle( QFrame::HLine | QFrame::Sunken );

   QVBoxLayout* groupLayout = new QVBoxLayout();
   groupLayout->addLayout(hVisVoxels);
   groupLayout->addWidget(separator0);
   groupLayout->addWidget(rbAtlasBinary);
   groupLayout->addWidget(cbMipmapping);
   groupLayout->addWidget(cbPrevox);
   groupLayout->addLayout(hboxLevel);
   groupLayout->addWidget(separator2);
   groupLayout->addWidget(cbMipmapTestVis);
   groupLayout->addWidget(cbShowVoxelCubes);
   groupLayout->addLayout(hboxCycle);
   groupLayout->addLayout(gbStartEnd);
   gbVisualizeVoxels->setLayout(groupLayout);

   QCheckBox* cbDisplayBB = new QCheckBox("Display Bounding Boxes");
   connect(cbDisplayBB, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleDisplayBoundingBoxes(bool)));
   cbDisplayBB->setChecked(SETTINGS->displayBoundingBoxes());

   QCheckBox* cbWorldAxes = new QCheckBox("Display World Space Axes");
   connect(cbWorldAxes, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleDisplayWorldSpaceAxes(bool)));

   QCheckBox* cbVoxelCam = new QCheckBox("Display Voxel Camera");
   connect(cbVoxelCam, SIGNAL(toggled(bool)), SETTINGS, SLOT(toggleDisplayVoxelCamera(bool)));

   // Put all GUI elements in one layout
   QVBoxLayout* vbox2 = new QVBoxLayout();
   vbox2->addWidget(gbVisualizeVoxels);
   vbox2->addWidget(cbVoxelCam);
   vbox2->addWidget(cbWorldAxes);
   vbox2->addWidget(cbDisplayBB); 

   vbox2->addStretch(1);

   layout->addLayout(vbox2);


}

void PageDebugging::toggleMipmapTestVisEnabled(bool enabled)
{
   cbShowVoxelCubes->setEnabled(enabled);
   cycleSlider->setEnabled(enabled);
   cycleIndexLabel->setEnabled(enabled);
   cycleSliderLabel->setEnabled(enabled);

   emit(toggledMipmapTestVisEnabled(enabled));
}


QGridLayout* PageDebugging::createStartEndPointControl()
{
   QDoubleSpinBox* spinStartX = new QDoubleSpinBox();
   QDoubleSpinBox* spinStartY = new QDoubleSpinBox();
   QDoubleSpinBox* spinStartZ = new QDoubleSpinBox();
   QDoubleSpinBox* spinEndX = new QDoubleSpinBox();
   QDoubleSpinBox* spinEndY = new QDoubleSpinBox();
   QDoubleSpinBox* spinEndZ = new QDoubleSpinBox();

   spinStartX->setSingleStep(0.1);
   spinStartY->setSingleStep(0.1);
   spinStartZ->setSingleStep(0.1);
   spinEndX->setSingleStep(0.1);
   spinEndY->setSingleStep(0.1);
   spinEndZ->setSingleStep(0.1);

   spinStartX->setDecimals(1);
   spinStartY->setDecimals(1);
   spinStartZ->setDecimals(1);
   spinEndX->setDecimals(1);
   spinEndY->setDecimals(1);
   spinEndZ->setDecimals(1);

   spinStartX->setMinimum(-5.0f);
   spinStartX->setMaximum( 5.0f);
   spinStartY->setMinimum(-5.0f);
   spinStartY->setMaximum( 5.0f);
   spinStartZ->setMinimum(-5.0f);
   spinStartZ->setMaximum( 5.0f);
   spinEndX->setMinimum(-5.0f);
   spinEndX->setMaximum( 5.0f);
   spinEndY->setMinimum(-5.0f);
   spinEndY->setMaximum( 5.0f);
   spinEndZ->setMinimum(-5.0f);
   spinEndZ->setMaximum( 5.0f);

   spinStartX->setToolTip("x-coordinate");
   spinStartY->setToolTip("y-coordinate");
   spinStartZ->setToolTip("z-coordinate");
   spinEndX->setToolTip("x-coordinate");
   spinEndY->setToolTip("y-coordinate");
   spinEndZ->setToolTip("z-coordinate");

   QGridLayout* grid = new QGridLayout();
   grid->addWidget(new QLabel("Start"), 0, 0);
   grid->addWidget(new QLabel("End"), 1, 0);
   grid->addWidget(spinStartX, 0, 1);
   grid->addWidget(spinStartY, 0, 2);
   grid->addWidget(spinStartZ, 0, 3);
   grid->addWidget(spinEndX, 1, 1);
   grid->addWidget(spinEndY, 1, 2);
   grid->addWidget(spinEndZ, 1, 3);

   // first: set default values defined by SETTINGS
   spinStartX->setValue(0.5f);
   spinStartY->setValue(0.1f);
   spinStartZ->setValue(0.4f);
   spinEndX->setValue(0.3f);
   spinEndY->setValue(0.6f);
   spinEndZ->setValue(0.1f);

   // next: connect GUI elements to Settings-slots
   connect(spinStartX, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setStartPointX(double)));
   connect(spinStartY, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setStartPointY(double)));
   connect(spinStartZ, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setStartPointZ(double)));
   connect(spinEndX, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setEndPointX(double)));
   connect(spinEndY, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setEndPointY(double)));
   connect(spinEndZ, SIGNAL(valueChanged(double)), SETTINGS, SLOT(setEndPointZ(double)));

   return grid;
}