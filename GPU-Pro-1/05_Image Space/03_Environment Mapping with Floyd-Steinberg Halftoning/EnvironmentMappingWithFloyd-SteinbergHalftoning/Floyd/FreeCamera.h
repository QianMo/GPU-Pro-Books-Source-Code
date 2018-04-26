#pragma once
#include "camera.h"
#include "DXUTCamera.h"

/// Class for non-enetity bound cameras. (Wraps a DXUT CFirstPersonCamera.)
class FreeCamera :
	public Camera
{
protected:
	/// DXUT camera instance.
	CFirstPersonCamera camera;
public:
	/// Returns eyePosition.
	const D3DXVECTOR3& getEyePosition();
	/// Returns the inverse of the view-projection matrix (without eye pos translation).
	const D3DXMATRIX& getOrientProjMatrixInverse();
	/// Returns view matrix.
	const D3DXMATRIX& getViewMatrix();
	/// Returns projection matrix.
	const D3DXMATRIX& getProjMatrix();

	/// Passes event to DXUT camera.
	void handleInput(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam);
	/// Animates DXUT camera.
	void animate(double dt);

	void setAspect(double aspect);

	/// Constructor. Initializes DXUT camera.
	FreeCamera(D3DXVECTOR3& eye, D3DXVECTOR3& lookAt);
};
