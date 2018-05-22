#include "trackball.h"
#include <GL/glut.h>


Trackball::Trackball(Camera& camera) :
camera_(camera),
performTracking(false)
{

}


void Trackball::mousePressEvent(int button, int x, int y) {
	if(button==GLUT_LEFT_BUTTON) {
		// start a trackball motion
		stopTracking();
		camera_.setViewMode(Camera::FromFixed);
		startTracking(x, y);
	}
	if(button==GLUT_MIDDLE_BUTTON) {
		// start a trackball motion
		stopTracking();
		camera_.setViewMode(Camera::CenterFixed);
		startTracking(x, y);
	}
}

void Trackball::mouseReleaseEvent(int button, int x, int y) {
	if(button==GLUT_LEFT_BUTTON || button==GLUT_MIDDLE_BUTTON) {
		stopTracking();
	}
}

void Trackball::mouseMoveEvent(int x, int y) {
	if(performTracking) {
		// left button pressed => Trackball
		int deltaX = x-trackballStartPosX_;
		int deltaY = y-trackballStartPosY_;
		float lat = trackballStartLat_+0.5*deltaY;
		if(lat<-88.0) lat = -88.0;
		if(lat>88.0) lat = 88.0;
		camera_.setLatitude(lat);
		camera_.setLongitude(trackballStartLong_+0.5*deltaX);

		glutPostRedisplay();
	}
}

void Trackball::startTracking(int x, int y) {
	if(performTracking) return;

	performTracking = true;
	trackballStartPosX_ = x;
	trackballStartPosY_ = y;
	trackballStartLat_ = camera_.latitude();
	trackballStartLong_ = camera_.longitude();
}

void Trackball::stopTracking() {
	performTracking = false;
}
