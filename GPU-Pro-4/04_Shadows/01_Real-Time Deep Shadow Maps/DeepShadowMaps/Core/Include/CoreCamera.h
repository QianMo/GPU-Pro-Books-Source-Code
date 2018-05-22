#pragma once

#include "CoreVector3.h"
#include "CoreMatrix4x4.h"
#include "CorePlane.h"
#include <d3d11.h>

// Default up vector is in y dir, default view, projection = unit matrix
class CoreCamera
{
public:
	CoreCamera();

	// Set perspective projection
	void SetProjectionPerspective(float fov, float aspect, float nearPlane, float farPlane);

	// Set parallel projection
	void SetProjectionParallel(float fov, float aspect, float nearPlane, float farPlane);

	// Set parallel projection
	void SetProjectionParallel(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

	// Sets an effect variable to the projection matrix
	void ProjectionToEffectVariable(ID3DX11EffectMatrixVariable* projectionVar);
	
	// Set the view
	void SetView(CoreVector3& pos, CoreVector3& lookAt, CoreVector3& up);

	// Set the view
	void SetView(CoreVector3& xAxis, CoreVector3& yAxis, CoreVector3& zAxis, CoreVector3& pos);

	// Get the view
	void GetView(CoreVector3& xAxis, CoreVector3& yAxis, CoreVector3& zAxis, CoreVector3& pos);
	void GetView(CoreVector3 &pos, CoreVector3& lookAt, CoreVector3 &up);

	// Sets an effect variable to the view matrix
	void ViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar);

	// Sets an effect variable to the inverted view matrix
	void InvViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar);

	// Sets an effect variable to the world*view*projection matrix
	void WorldViewProjectionToEffectVariable(ID3DX11EffectMatrixVariable* viewVar, CoreMatrix4x4 &world);

	// Sets an effect variable to the world*view matrix
	void WorldViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar, CoreMatrix4x4 &world);

	// Moves the cam forward in view direction
	void GoForward(float distance);

	// Moves the cam backward to view direction
	void GoBackward(float distance);

	// Moves the cam left to view direction
	void GoLeft(float distance);

	// Moves the cam right to view direction
	void GoRight(float distance);

	void GoUp(float distance);
	void GoDown(float distance);

	// MouseLook function
	void MouseLook(float deltaX, float deltaY);

	// Inline getters
	inline CoreMatrix4x4 GetProjection() { return projection; }
	inline CoreMatrix4x4 GetView()		 { return view; }
	inline CoreVector3   GetPosition()   { return pos; }
	inline CoreVector3	 GetXAxis()		 { return xAxis; }
	inline CoreVector3	 GetYAxis()		 { return yAxis; }
	inline CoreVector3	 GetZAxis()		 { return zAxis; }

	CoreMatrix4x4 GetClipProjectionMatrix(CorePlane &ClipPlane);

	float GetNearClip() { return m_nearPlane; }
	float GetFarClip() { return m_farPlane; }

private:
	enum {
		TOP = 0,
		BOTTOM,
		LEFT,
		RIGHT,
		NEARP,
		FARP
	};


	CoreMatrix4x4 projection;
	CoreMatrix4x4 view;
	CoreVector3 pos, xAxis, yAxis, zAxis, up;	// up vector needed for mouselook

	float m_nearPlane, m_farPlane;
	float nw, nh, fw, fh;		// Near plane width,...
	CorePlane frustumPlanes[6];

	void generateViewFromPrivateVariables();
	void calcFrustumData();
};