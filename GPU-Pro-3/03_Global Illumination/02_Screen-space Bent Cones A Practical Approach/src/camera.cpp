#include "camera.h"

#include <Gl/glew.h>

#include <glm/gtc/matrix_projection.hpp>
#include <glm/gtx/transform2.hpp>

using namespace glm;

Camera::Camera() {
    fovy_ = 60.0f;
    nearPlane_ = 0.1f;
    farPlane_ = 2000.0f;

    longitude_ = 180.0;
    latitude_ = 45.0;
    distance_ = 1.0;
    viewMode_ = CenterFixed;
}

void Camera::init(unsigned int width, unsigned int height) {
    setViewportAspectRatio(width, height);

    updateModelView();
    updateProjection();
}

void Camera::setViewMode(Mode mode) {
    if(viewMode_==ViewFixed && mode!=ViewFixed) {
        // translate current direction to lat/lon/distance
        vec3 dir = from_-center_;
        distance_ = length(dir);
        longitude_ = degrees(atan2(dir[0], dir[1]));
        latitude_ = degrees(asin(dir[2]/distance_));
    }
    viewMode_ = mode;
}

void Camera::move(const vec3 &d) {
    center_ += d;
    from_ += d;
    updateModelView();
}

void Camera::setView(const vec3 &pos, const vec3 &target) {
    // use frames
    setViewMode(ViewFixed);

    // store position
    from_ = pos;
    center_ = target;
    distance_ = glm::length(center_-from_);

    updateModelView();
}

void Camera::setLongitude(float l) {
    longitude_ = l;
    updateModelView();
}

void Camera::setLatitude(float l) {
    latitude_ = l;
    updateModelView();
}

void Camera::setDistance(float d) {
    setViewMode(CenterFixed);
    distance_ = d;
    updateModelView();
}

void Camera::setCenter(const vec3 &c) {
    setViewMode(CenterFixed);
    center_ = c;
    updateModelView();
}

void Camera::setFrom(const vec3 &f) {
    setViewMode(FromFixed);
    from_ = f;
    updateModelView();
}

void Camera::setViewportAspectRatio(unsigned int width, unsigned int height) {
	setViewportAspectRatio(float(width) / float(height));
}

void Camera::setViewportAspectRatio(float ratio) {
	viewportAspectRatio_ = ratio;
	updateProjection();
}

void Camera::setNearPlane(float nearPlane) {
    nearPlane_ = nearPlane;
    updateProjection();
}

void Camera::setFarPlane(float farPlane) {
    farPlane_ = farPlane;
    updateProjection();
}

void Camera::setFovY(float fov) {
    fovy_ = fov;
    updateProjection();
}

float Camera::longitude() const {
    return longitude_;
}

float Camera::latitude() const {
    return latitude_;
}

float Camera::distance() const {
    return distance_;
}

glm::vec3 Camera::center() const {
    return center_;
}

glm::vec3 Camera::from() const {
    return from_;
}

float Camera::viewportAspectRatio() const {
	return viewportAspectRatio_;
}

float Camera::nearPlane() const {
    return nearPlane_;
}

float Camera::farPlane() const {
    return farPlane_;
}

vec3 Camera::direction() const {
    return center_-from_;
}

float Camera::fovY() const {
    return fovy_;
}

const glm::mat4& Camera::MVP() const {
    return MVP_;
}

const glm::mat4& Camera::projectionM() const {
    return projectionM_;
}

const glm::mat4& Camera::invProjectionM() const {
    return invProjectionM_;
}

const glm::mat4& Camera::modelView() const {
    return modelViewM_;
}

const glm::mat3& Camera::normalM() const {
    return normalM_;
}

void Camera::updateProjection() {
    projectionM_ = perspective(fovy_, viewportAspectRatio_, nearPlane_, farPlane_);
    invProjectionM_ = inverse(projectionM_);
    MVP_ = projectionM_ * modelViewM_;

    projectionChanged_ = true;
}

void Camera::updateModelView() {
    switch (viewMode_) {
        case FromFixed:
        {
            // use latlon for view definition
            float longR = radians(longitude_);
            float latR = radians(latitude_);
            center_[0] = from_[0] - cos(latR)*sin(longR)*distance_;
            center_[1] = from_[1] - cos(latR)*cos(longR)*distance_;
            center_[2] = from_[2] - sin(latR)*distance_;
            break;
        }
        case CenterFixed:
        {
            // use latlon for view definition
            float longR = radians(longitude_);
            float latR = radians(latitude_);
            from_[0] = cos(latR)*sin(longR)*distance_ + center_[0];
            from_[1] = cos(latR)*cos(longR)*distance_ + center_[1];
            from_[2] = sin(latR)*distance_ + center_[2];
            break;
        }
        default:
            break;
    }
    modelViewM_ = lookAt(from_, center_, vec3(0.0, 0.0, 1.0));
    normalM_ = mat3(inverse(modelViewM_));
    normalM_ = transpose(normalM_);

    MVP_ = projectionM_ * modelViewM_;

    modelViewChanged_ = true;
}

bool Camera::modelViewChanged() const {
    return modelViewChanged_;
}

bool Camera::projectionChanged() const {
    return projectionChanged_;
}

void Camera::resetModelViewChanged() {
    modelViewChanged_ = false;
}

void Camera::resetProjectionChanged() {
    projectionChanged_ = false;
}

