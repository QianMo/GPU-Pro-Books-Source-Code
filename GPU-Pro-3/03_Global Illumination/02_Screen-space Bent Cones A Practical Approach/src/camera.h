#pragma once

#include <glm/setup.hpp>
#define GLM_USE_ONLY_XYZW
#include <glm/glm.hpp>

/*
The camera is defined using a focus point (center) and a camera location
defined as polar coordinates (longitude, latitude, distance) 
relative to that focus point.
*/

class Camera  {

public:
    Camera();
    void init(unsigned int width, unsigned int height);

    void move(const glm::vec3 &d);

    //////////////////////////////////////////////////////////////////////////
    // matrices
    const glm::mat4& MVP() const;
    const glm::mat4& projectionM() const;
    const glm::mat4& invProjectionM() const;
    const glm::mat4& modelView() const;
    const glm::mat3& normalM() const;

    bool modelViewChanged() const;
    bool projectionChanged() const;

    void resetModelViewChanged();
    void resetProjectionChanged();

    //////////////////////////////////////////////////////////////////////////
    // params

    // getter
    float longitude() const;
    float latitude() const;
    float distance() const;
    glm::vec3 center() const;
    glm::vec3 from() const;
    glm::vec3 direction() const;

    float fovY() const;
    float nearPlane() const;
    float farPlane() const;
	float viewportAspectRatio() const;

    // setter
    void setFovY(float fovy);
	void setViewportAspectRatio(unsigned int width, unsigned int height);
	void setViewportAspectRatio(float ratio);
    void setNearPlane(float nearPlane);
    void setFarPlane(float farPlane);

    void setLongitude(float l);
    void setLatitude(float l);
    void setDistance(float d);
    void setCenter(const glm::vec3 &c);
    void setFrom(const glm::vec3 &f);
    void setView(const glm::vec3 &pos, const glm::vec3 &target);

    enum Mode { FromFixed, CenterFixed, ViewFixed };
    void setViewMode(Mode mode);

private:
    void updateProjection();
    void updateModelView();

private:
    float longitude_, latitude_, distance_;
    glm::vec3 center_;
    glm::vec3 from_;

    float fovy_;
    float nearPlane_;
    float farPlane_;
	float viewportAspectRatio_;

    glm::mat4 MVP_;
    glm::mat4 projectionM_;
    glm::mat4 invProjectionM_;
    glm::mat4 modelViewM_;
    glm::mat3 normalM_;

    bool modelViewChanged_;
    bool projectionChanged_;

    Mode viewMode_;
};
