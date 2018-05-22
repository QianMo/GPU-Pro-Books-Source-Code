#ifndef __LIGHT__H__
#define __LIGHT__H__

#include "../Util/Matrix4.h"
#include "../Util/Color.h"

class Light
{
public:
	Light(void);
	~Light(void);

	/// Inits the light
	void Init(const Vector3 position=Vector3(0.0f, 0.0f, 0.0f), const Vector3 lookAt=Vector3(0.0f, 0.0f, 0.0f));

	/// Updates the light
	void Update(const Vector3 position, const Vector3 lookAt, float _fieldOfView, float aspectRation, float _nearPlane, float _farPlane);

	/// Render lights frustum
	void DebugRender(void);

	/// Teturns the view matrix of the light
	Matrix4 GetViewMatrix(void) const { return lightView; }

	/// Teturns the projection matrix of the light
	Matrix4 GetProjectionMatrix(void) const { return lightProjection; }

	/// Teturn position stuff
	const Vector3& GetLightPosition(void) const { return lightPosition; }
	const Vector3& GetLightLookAt(void) const { return lightLookAt; }
	float GetNearPlane(void) const { return nearPlane; }
	float GetFarPlane(void) const { return farPlane; }
	float GetFieldOfView(void) const { return fieldOfView; }

	/// Returns the diffuse color of the light
	const Color GetDiffuseColor(void) const { return lightDiff; }
	void SetDiffuseColor(Color diff) { lightDiff = diff; }

	void SetLightRadiusWorld(float radius);
	float GetLightRadiusWorld(void) const { return lightRadiusWorld; }

private:
	Vector3 lightPosition;
	Vector3 lightLookAt;

	float nearPlane;
	float farPlane;

	float fieldOfView;

	float lightRadiusWorld;

	Matrix4 lightProjection;
	Matrix4 lightView;

	Color lightDiff;
};

#endif