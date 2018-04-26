#pragma once

/// Basic camera interface, to be implemented by camera type classes.
class Camera
{
public:
	/// Returns eye position.
	virtual const D3DXVECTOR3& getEyePosition()=0;
	/// Returns the ahead vector.
	D3DXVECTOR3 getAhead();
	/// Returns the inverse of the view-projection matrix (without eye pos translation) to be used in shaders.
	virtual const D3DXMATRIX& getOrientProjMatrixInverse()=0;
	/// Returns view matrix to be used in shaders.
	virtual const D3DXMATRIX& getViewMatrix()=0;
	/// Returns projection matrix to be used in shaders.
	virtual const D3DXMATRIX& getProjMatrix()=0;

	/// Manipulates camera. To be implemented if the camera is directly controlled by the user.
	virtual void handleInput(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam){}
	/// Moves camera. To be implemented if the camera has its own animation mechanism.
	virtual void animate(double dt){}

	virtual void setAspect(double aspect)=0;
};
