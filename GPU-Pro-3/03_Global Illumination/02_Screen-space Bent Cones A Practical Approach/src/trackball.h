#pragma once

#include "camera.h"

class MouseInputHandler {
public:
	virtual void mousePressEvent(int button, int x, int y) = 0;
	virtual void mouseReleaseEvent(int button, int x, int y) = 0;
	virtual void mouseMoveEvent(int x, int y) = 0;
};

class Trackball : public MouseInputHandler {
public:
	Trackball(Camera& camera);

	virtual void mousePressEvent(int button, int x, int y);
	virtual void mouseReleaseEvent(int button, int x, int y);
	virtual void mouseMoveEvent(int x, int y);

private:
	void startTracking(int x, int y);
	void stopTracking();

private:
	Camera& camera_;

	int trackballStartPosX_, trackballStartPosY_;
	double trackballStartLong_, trackballStartLat_;
	bool performTracking;
};