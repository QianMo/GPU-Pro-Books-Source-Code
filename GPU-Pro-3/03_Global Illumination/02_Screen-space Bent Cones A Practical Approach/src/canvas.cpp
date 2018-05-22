#include "canvas.h"
#include <cmath>
#include <sstream>
#include <glm/glm.hpp>
#include <ctime>

using namespace glm;

namespace {
	const std::string modelFile = "mushroom_field.obj";
	const std::string modelDir = ".\\meshes\\";
	
	const bool flipYZ = true;
	const float scaleModel = 1.0f;


	// helper
	std::ostream& operator<<(std::ostream& out, const glm::mat4& mat) {
		out.precision(4);
		out.setf(std::ios::fixed, std::ios::floatfield);
		for(size_t i=0; i<4; ++i) {
			for(size_t j=0; j<4; ++j) {
				out.width(8);
				out << mat[i][j];
			}
			out << std::endl;
		}
		return out;
	}
}

Canvas::Canvas(unsigned int width, unsigned int height)
: width_(width), height_(height),
quad_(Quad::InstanceRef()),
mouseInputHandler_(new Trackball(camera_)),
gammaCorrection_(1.0f),
bnTechnique_(NULL), bnBlur_(NULL), envMapRenderer_(NULL), envMapConeRenderer_(NULL), clTechnique_(NULL), skyRenderer_(NULL),
timerMode_(Nothing),
frameCountForAvgTime_(50),
millisecondsUntilTimerPrint(2000)
{

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	// parameters for algorithm
	sSBCParameters_.sampleRadius = 1.0f;
	sSBCParameters_.maxDistance = sSBCParameters_.sampleRadius * 1.6;
	sSBCParameters_.numRayMarchingSteps = 3;
	sSBCParameters_.patternSize = 8;
	sSBCParameters_.rayMarchingBias = sSBCParameters_.sampleRadius / float(sSBCParameters_.numRayMarchingSteps) / 1000.0f;
	sSBCParameters_.sampleCount = 8;

	blurParameters_.kernelSize = sSBCParameters_.patternSize;
	blurParameters_.positionPower = 2.0;
	blurParameters_.normalPower = 5.0;
	blurParameters_.subSampling = 1;

	//////////////////////////////////////////////////////////////////////////
	// some lights in the env map
	lights_.lights.push_back(LightEnvMapRenderer::InputParameters::Light(vec3(10.0, 8.0, 0.6), vec3(1.0, 1.0, 0.22), M_PI * 0.5f * 0.08, M_PI * 0.5f * 0.1));
	lights_.lights.push_back(LightEnvMapRenderer::InputParameters::Light(vec3(0.3, 0.4, 0.5), vec3(0.0, 0.0, 1.0), M_PI * 0.5f * 0.7, M_PI * 0.5f * 0.5));
	lights_.lights.push_back(LightEnvMapRenderer::InputParameters::Light(vec3(0.6, 0.7, 7.5), vec3(-1.0, -1.0, 0.5), M_PI * 0.5f * 0.1, M_PI * 0.5f * 0.1));

	convolutionParameters_.sampleCount = 128;

	cubeMapResolution_ = 128;
	conesCount_ = 8;

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	allTimers_.push_back(&GBTimer_);
	allTimers_.push_back(&SSBCTimer_);
	allTimers_.push_back(&blurTimer_);
	allTimers_.push_back(&coneLightingTimer_);
	allTimers_.push_back(&frameTimer_);
	historyTimerValues_.resize(5);
}

Canvas::~Canvas() {
	delete bnTechnique_;
	delete bnBlur_;
	delete envMapRenderer_;
	delete envMapConeRenderer_;
	delete skyRenderer_;
    cleanupGL();
	delete mouseInputHandler_;
}

void Canvas::cleanupGL() {
	quad_.cleanup();
	gbuffer_.cleanup();

	delete bnTechnique_;
	bnTechnique_ = NULL;
	delete bnBlur_;
	bnBlur_ = NULL;
	delete envMapRenderer_;
	envMapRenderer_ = NULL;
	delete envMapConeRenderer_;
	envMapConeRenderer_ = NULL;
	delete clTechnique_;
	clTechnique_ = NULL;
	delete skyRenderer_;
	skyRenderer_ = NULL;

	for(size_t i=0; i<allTimers_.size(); ++i) {
		allTimers_[i]->cleanup();
	}
}

void Canvas::paintGL() {
	if(timerMode_ == Frame) frameTimer_.startTimer();
	GLCHECK(glClear(GL_COLOR_BUFFER_BIT));

	const glm::mat4& MVP = camera_.MVP();
	const glm::mat3& normalM = camera_.normalM();

	if(timerMode_ == Techniques) GBTimer_.startTimer();
	gbuffer_.preRender(&MVP[0][0], &normalM[0][0], gammaCorrection_);
	scene_.renderGeometry(gbuffer_);
	gbuffer_.postRender();
	if(timerMode_ == Techniques) GBTimer_.stopTimer();

	// only pixel operations ahead
	Quad::InstanceRef().preRender();

	if(timerMode_ == Techniques) SSBCTimer_.startTimer();
	bnTechnique_->render(camera_.modelView(), camera_.MVP(), &gbuffer_);
	if(timerMode_ == Techniques) SSBCTimer_.stopTimer();

	if(timerMode_ == Techniques) blurTimer_.startTimer();
	bnBlur_->render(bnTechnique_->output(), &gbuffer_);
	if(timerMode_ == Techniques) blurTimer_.stopTimer();

	if(timerMode_ == Techniques) coneLightingTimer_.startTimer();
	clTechnique_->render(gbuffer_, bnBlur_->output(), envMapConeRenderer_->output(), conesCount_);
	if(timerMode_ == Techniques) coneLightingTimer_.stopTimer();

	Quad::InstanceRef().postRender();

	//////////////////////////////////////////////////////////////////////////

	clTechnique_->output().bindReadFBO();
	GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
	GLCHECK(glBlitFramebuffer(0, 0, clTechnique_->output().width(), clTechnique_->output().height(), 0, 0, width_, height_, GL_COLOR_BUFFER_BIT, GL_NEAREST));
	gbuffer_.unbindReadFBO();

	gbuffer_.bindReadFBO();
	GLCHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
	GLCHECK(glBlitFramebuffer(0, 0, gbuffer_.width(), gbuffer_.height(), 0, 0, width_, height_, GL_DEPTH_BUFFER_BIT, GL_NEAREST));
	gbuffer_.unbindReadFBO();

	skyRenderer_->render(inverse(camera_.MVP()), envMapRenderer_->output(), width_, height_);
	if(timerMode_ == Frame) frameTimer_.stopTimer();

	updateTimerResults();
	printTimerResults();
}

void Canvas::resizeGL(int width, int height) {
	if(width == 0 && height == 0) return;
	camera_.setViewportAspectRatio(width, height);
    gbuffer_.resize(width, height);
	bnTechnique_->resize(width, height);
	bnBlur_->resize(width, height);
	clTechnique_->resize(width, height);

	width_ = width;
	height_ = height;
}

void Canvas::initializeGL() {
	for(size_t i=0; i<allTimers_.size(); ++i) {
		allTimers_[i]->init();
	}

	GLCHECK(glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
	GLCHECK(glClearDepth(1.0f));

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	quad_.init();
    quad_.preRender();

	gbuffer_.init(width_, height_);

	initCamera();

	scene_.init(modelDir, modelFile, flipYZ, scaleModel);

	bnTechnique_ = new BentNormalRayMarchingTechnique;
	bnTechnique_->setStaticParameters(sSBCParameters_);
	bnTechnique_->resize(width_, height_);
	bnBlur_ = new BentNormalBlurring;
	bnBlur_->setStaticParameters(blurParameters_);
	bnBlur_->resize(width_,height_);

	envMapRenderer_ = new LightEnvMapRenderer;
	envMapRenderer_->resize(cubeMapResolution_);
	envMapConeRenderer_ = new EnvMapConvolution;
	envMapConeRenderer_->resize(cubeMapResolution_, conesCount_);

	envMapRenderer_->render(lights_);
	envMapConeRenderer_->render(convolutionParameters_, envMapRenderer_->output());

	clTechnique_ = new ConeLightingTechnique;

	skyRenderer_ = new SkyRenderer;

	loadShader();
}

void Canvas::loadShader() {
	gbuffer_.reloadShader();
}

void Canvas::initCamera() {
	walkSpeed = 1.0;
	upSpeed = 0.2;

	camera_.setFovY(60.0f);
	camera_.setNearPlane(1.0f);
	camera_.setFarPlane(100.0f);

	camera_.setView(glm::vec3(-15.0, 15.0, 10.0), glm::vec3(-2.0, 2.0, -7.0));
	
	camera_.init(width_, height_);
}

void Canvas::keyPressEvent(unsigned char c, int x, int y) {
	switch(c) {

	case 'r': {
		//loadShader();
		break;
			  }
	case '1': {
		sSBCParameters_.sampleRadius *= 0.5f;
		sSBCParameters_.maxDistance = sSBCParameters_.sampleRadius * 1.6f;
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO Radius: " << sSBCParameters_.sampleRadius << std::endl;
		break;
			  }
	case '2': {
		sSBCParameters_.sampleRadius *= 2.0f;
		sSBCParameters_.maxDistance = sSBCParameters_.sampleRadius * 1.6f;
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO Radius: " << sSBCParameters_.sampleRadius << std::endl;
		break;
			  }
	case '3': {
		sSBCParameters_.sampleCount = max(1u, sSBCParameters_.sampleCount / 2);
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO sample count per pixel: " << sSBCParameters_.sampleCount << std::endl;
		break;
			  }
	case '4': {
		sSBCParameters_.sampleCount = max(1u, sSBCParameters_.sampleCount * 2);
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO sample count per pixel: " << sSBCParameters_.sampleCount << std::endl;
		break;
			  }
	case '5': {
		sSBCParameters_.numRayMarchingSteps = max(1u, sSBCParameters_.numRayMarchingSteps - 1);
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO ray marching steps per sample: " << sSBCParameters_.numRayMarchingSteps << std::endl;
		break;
			  }
	case '6': {
		sSBCParameters_.numRayMarchingSteps += 1u;
		bnTechnique_->setStaticParameters(sSBCParameters_);
		std::cout << "AO ray marching steps per sample: " << sSBCParameters_.numRayMarchingSteps << std::endl;
		break;
			  }

	//////////////////////////////////////////////////////////////////////////
	// super sampling does not work with blit
	//case '5': {
	//	blurParameters_.subSampling = max(1, blurParameters_.subSampling / 2);
	//	std::cout << "Super-sampling: " << blurParameters_.subSampling << std::endl;
	//	gbuffer_.resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	bnTechnique_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	bnBlur_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	clTechnique_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	break;
	//		  }
	//case '6': {
	//	blurParameters_.subSampling = min(8, blurParameters_.subSampling * 2);
	//	std::cout << "Super-sampling: " << blurParameters_.subSampling << std::endl;
	//	gbuffer_.resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	bnTechnique_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	bnBlur_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	clTechnique_->resize(width_ * blurParameters_.subSampling, height_ * blurParameters_.subSampling);
	//	break;
	//		  }
	case 't': {
		clearOldTimerValues();
		timerMode_ = CaptureRenderingTimeMode(int(timerMode_)+1 % 3);
		break;
			  }
	case 'w': {
		glm::vec3 dir = camera_.direction();
		dir[2] = 0.0;
		dir = glm::normalize(dir) * walkSpeed;
		camera_.move(dir);
		break;
			  }
	case 's': {
		glm::vec3 dir = camera_.direction();
		dir[2] = 0.0;
		dir = glm::normalize(-dir) * walkSpeed;
		camera_.move(dir);
		break;
			  }
	case 'a': {
		glm::vec3 dir = camera_.direction();
		glm::vec3 dir2(-dir[1], dir[0], 0.0);
		dir2 = glm::normalize(dir2) * walkSpeed;
		camera_.move(dir2);
		break;
			  }
	case 'd': {
		glm::vec3 dir = camera_.direction();
		glm::vec3 dir2(dir[1], -dir[0], 0.0);
		dir2 = glm::normalize(dir2) * walkSpeed;
		camera_.move(dir2);
		break;
			  }
	case '+': {
		float dist = camera_.distance() * 0.9f;
		if(dist<0.01) dist = 0.01;
		camera_.setDistance(dist);
		break;
			  }
	case '-': {
		float dist = camera_.distance() * 1.12f;
		if(dist>10000.0) dist = 10000.0f;
		camera_.setDistance(dist);
		break;
			  }
	case 'q': {
		camera_.move(glm::vec3(0.0, 0.0, upSpeed));
		break;
			  }
	case 'e': {
		camera_.move(glm::vec3(0.0, 0.0, -upSpeed));
		break;
			  }

	}
}

void Canvas::printTimerResults() {
	static clock_t lastTime = clock();
	clock_t now = clock();
	clock_t timeElapsed = (now - lastTime) / CLOCKS_PER_SEC * 1000;
	if(timeElapsed < millisecondsUntilTimerPrint) {
		return;
	}

	switch(timerMode_) {
		case Techniques: {
			std::vector<float> avgValues(historyTimerValues_.size()-1);

			for(size_t i=0; i<avgValues.size(); ++i) {
				avgValues[i] = 0.0f;
				for(std::list<float>::const_iterator j=historyTimerValues_[i].begin(); j!=historyTimerValues_[i].end(); ++j) {
					avgValues[i] += *j / float(historyTimerValues_[i].size());
				}
			}
			std::cout << "GBuffer: " << avgValues[0] << "ms" << std::endl;
			std::cout << "Bent Normal & SSAO: " << avgValues[1] << "ms" << std::endl;
			std::cout << "Bent Normal & SSAO Blur: " << avgValues[2] << "ms" << std::endl;
			std::cout << "Cone computation & single lookup: " << avgValues[3] << "ms" << std::endl;
			break;
						 }
		case Frame: {
			float avgValue = 0.0f;

			for(std::list<float>::const_iterator j=historyTimerValues_.back().begin(); j!=historyTimerValues_.back().end(); ++j) {
				avgValue += *j / float(historyTimerValues_.back().size());
			}
			std::cout << "Total frame: " << avgValue << "ms" << std::endl;
			break;
					}
		case Nothing:
		default:;
	}
	lastTime = clock();
}

void Canvas::clearOldTimerValues() {
	switch(timerMode_) {
		case Techniques: {
			for(size_t i=0; i<historyTimerValues_.size()-1; ++i) {
				historyTimerValues_[i].clear();
			}
			break;
						 }
		case Frame: {
			historyTimerValues_.back().clear();
			break;
					}
		case Nothing:
		default:
			for(size_t i=0; i<historyTimerValues_.size(); ++i) {
				historyTimerValues_[i].clear();
			}
	}
}

void Canvas::updateTimerResults() {
	switch(timerMode_) {
		case Techniques: {
			for(size_t i=0; i<historyTimerValues_.size()-1; ++i) {
				historyTimerValues_[i].push_back(allTimers_[i]->getLatestTime());
				if(historyTimerValues_[i].size() > frameCountForAvgTime_) historyTimerValues_[i].pop_front();
			}
			break;
						 }
		case Frame: {
			historyTimerValues_.back().push_back(frameTimer_.getLatestTime());
			if(historyTimerValues_.back().size() > frameCountForAvgTime_) historyTimerValues_.back().pop_front();
			break;
					}
		case Nothing:
		default:;
	}
}
