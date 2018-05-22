#pragma once

// OpenGL
#include "glhelper.h"

// STL
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include <NV/GPU_timer.h>

#include "single_texture2d_rt.h"
#include "quad.h"
#include "program.h"

#include "scene_renderer.h"
#include "gbuffer.h"
#include "bent_normal_ray_marching_technique.h"
#include "bent_normal_blurring.h"

#include "light_env_map_renderer.h"
#include "env_map_convolution.h"
#include "cone_lighting_technique.h"

#include "sky_renderer.h"

#include "trackball.h"

class Canvas {
public:
    Canvas(unsigned int width, unsigned int height);
    ~Canvas();

    virtual void paintGL();
    void resizeGL(int width, int height);
    void initializeGL();

	void keyPressEvent(unsigned char c, int x, int y);

	MouseInputHandler& mouseInputHandler() { return *mouseInputHandler_; }

	inline bool idleRedraw() const { return true; };

private:
    void cleanupGL();
    void loadShader();

	Camera camera_;
	MouseInputHandler* mouseInputHandler_;
	void initCamera();

	void updateTimerResults();
	void printTimerResults();
	void clearOldTimerValues();

private:
    unsigned int width_;
    unsigned int height_;

	SceneRenderer scene_;
	GBuffer gbuffer_;
    Quad& quad_;

	float gammaCorrection_;

    BentNormalRayMarchingTechnique* bnTechnique_;
	BentNormalRayMarchingTechnique::InputParameters sSBCParameters_;
	BentNormalBlurring* bnBlur_;
	BentNormalBlurring::InputParameters blurParameters_;

	ConeLightingTechnique* clTechnique_;

	LightEnvMapRenderer* envMapRenderer_;
	LightEnvMapRenderer::InputParameters lights_;
	EnvMapConvolution* envMapConeRenderer_;
	EnvMapConvolution::InputParameters convolutionParameters_;
	int cubeMapResolution_;
	int conesCount_;

	SkyRenderer* skyRenderer_;

	enum CaptureRenderingTimeMode {
		Nothing,
		Techniques,
		Frame
	};

	CaptureRenderingTimeMode timerMode_;

	GPUtimer GBTimer_;
	GPUtimer SSBCTimer_;
	GPUtimer blurTimer_;
	GPUtimer coneLightingTimer_;
	GPUtimer frameTimer_;

	std::vector<GPUtimer*> allTimers_;
	std::vector<std::list<float>> historyTimerValues_;
	size_t frameCountForAvgTime_;
	long millisecondsUntilTimerPrint;

	float walkSpeed;
	float upSpeed;
};