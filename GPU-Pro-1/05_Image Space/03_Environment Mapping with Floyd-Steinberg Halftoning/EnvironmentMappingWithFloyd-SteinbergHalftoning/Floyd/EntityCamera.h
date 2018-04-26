#pragma once
#include "camera.h"

class Entity;

/// Camera class for entity-bound cameras.
class EntityCamera :
	public Camera
{
	/// Owner entity.
	Entity* owner;

	// relative to entity:
	/// Camera position.
	D3DXVECTOR3 eyePosition;
	/// Camera position.
	D3DXVECTOR3 lookAt;
	/// Camera up vector.
	D3DXVECTOR3 up;
	/// Camera orientation matrix relative to entity.
	D3DXMATRIX viewMatrix;
	/// Projection matrix.
	D3DXMATRIX projMatrix;

	double fov;
	double aspect;
	double front;
	double back;

	// in world space:
	/// Camera position in world space.
	D3DXVECTOR3 worldEyePosition;
	/// View matrix.
	D3DXMATRIX worldViewMatrix;
	/// Inverse of view-projection matrix (without eye translation).
	D3DXMATRIX worldOrientProjMatrixInverse;
public:
	/// Returns camera position.
	const D3DXVECTOR3& getEyePosition();
	/// Returns view-projection matrix.
	const D3DXMATRIX& getOrientProjMatrixInverse();
	/// Returns view matrix.
	const D3DXMATRIX& getViewMatrix();
	/// Returns projection matrix.
	const D3DXMATRIX& getProjMatrix();

	/// Constructor. Places camera to entity origin, uses default perspective.
	EntityCamera(Entity* owner);
	/// Constructor. Uses default perspective.
	EntityCamera(Entity* owner, const D3DXVECTOR3& eyePosition, const D3DXVECTOR3& lookAt, const D3DXVECTOR3& up);
	/// Constructor.
	EntityCamera(Entity* owner, const D3DXVECTOR3& eyePosition, const D3DXVECTOR3& lookAt, const D3DXVECTOR3& up, double fov, double aspect, double front, double back);

	void setAspect(double aspect);
};
