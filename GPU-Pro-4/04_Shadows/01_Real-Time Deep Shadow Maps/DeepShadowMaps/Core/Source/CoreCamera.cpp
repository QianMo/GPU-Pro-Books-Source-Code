#include "Core.h"

CoreCamera::CoreCamera()
{
	up.y = 1.0f;	// Default up vector is in y dir
}

// Set perspective projection
void CoreCamera::SetProjectionPerspective(float fov, float aspect, float nearPlane, float farPlane)
{
	float height = fov / 2;
	height = cosf(height) / sinf(height);
	float width = height / aspect;

	m_nearPlane = nearPlane;
	m_farPlane = farPlane;

	float tang = tanf(fov / 2) ;
	nh = nearPlane * tang;
	nw = nh * aspect; 
	fh = farPlane  * tang;
	fw = fh * aspect;

	projection = CoreMatrix4x4(width,   0.0f,                                              0.0f, 0.0f,
							    0.0f, height,                                              0.0f, 0.0f,
							    0.0f,   0.0f,                 farPlane / (farPlane - nearPlane), 1.0f,
							    0.0f,   0.0f,  (-nearPlane) * farPlane / (farPlane - nearPlane), 0.0f );
	calcFrustumData();
}

// Set parallel projection
void CoreCamera::SetProjectionParallel(float fov, float aspect, float nearPlane, float farPlane)
{
	float height = fov / 2;
	height = cosf(height) / sinf(height);
	float width = height / aspect;

	m_nearPlane = nearPlane;
	m_farPlane = farPlane;

	float tang = tanf(fov / 2);
	nh = nearPlane * tang;
	nw = nh * aspect; 
	fh = nh;
	fw = nw;

	projection = CoreMatrix4x4(width,   0.0f,                                              0.0f, 0.0f,
							    0.0f, height,                                              0.0f, 0.0f,
							    0.0f,   0.0f,								    1.0f / farPlane, 0.0f,
							    0.0f,   0.0f,							  -nearPlane / farPlane, 1.0f );
	calcFrustumData();
}

// Set the view
void CoreCamera::SetView(CoreVector3& pos, CoreVector3& lookAt, CoreVector3& up)
{
	zAxis = CoreVector3(CoreVector3Normalize(lookAt - pos));
	xAxis = CoreVector3(CoreVector3Normalize(CoreVector3Cross(up, zAxis)));
	yAxis = CoreVector3(CoreVector3Normalize(CoreVector3Cross(zAxis, xAxis)));
	this->pos = pos;
	this->up = up;

	generateViewFromPrivateVariables();
}


// Sets an effect variable to the projection matrix
void CoreCamera::ProjectionToEffectVariable(ID3DX11EffectMatrixVariable* projectionVar)
{
	projectionVar->SetMatrix(projection.arr);
}

// Sets an effect variable to the view matrix
void CoreCamera::ViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar)
{
	viewVar->SetMatrix(view.arr);
}

// Sets an effect variable to the inverted view matrix
void CoreCamera::InvViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar)
{
	viewVar->SetMatrix(view.Invert().arr);
}

// Moves the cam forward in view direction
void CoreCamera::GoForward(float distance)
{
	CoreVector3 delta = zAxis * distance;

	pos += delta;
	generateViewFromPrivateVariables();
}

// Moves the cam backward to view direction
void CoreCamera::GoBackward(float distance)
{
	CoreVector3 delta = zAxis * distance;

	pos -= delta;
	generateViewFromPrivateVariables();
}

// Moves the cam left to view direction
void CoreCamera::GoLeft(float distance)
{
	CoreVector3 delta = xAxis * distance;

	pos -= delta;
	generateViewFromPrivateVariables();
}

// Moves the cam right to view direction
void CoreCamera::GoRight(float distance)
{
	CoreVector3 delta = xAxis * distance;

	pos += delta;
	generateViewFromPrivateVariables();
}

void CoreCamera::GoUp(float distance)
{
	pos.y += distance;
	generateViewFromPrivateVariables();
}

void CoreCamera::GoDown(float distance)
{
	pos.y -= distance;
	generateViewFromPrivateVariables();
}

// Set the view
void CoreCamera::SetView(CoreVector3& xAxis, CoreVector3& yAxis, CoreVector3& zAxis, CoreVector3& pos)
{
	this->xAxis = xAxis;
	this->yAxis = yAxis;
	this->zAxis = zAxis;
	this->pos = pos;

	generateViewFromPrivateVariables();	
}

void CoreCamera::GetView(CoreVector3& xAxis, CoreVector3& yAxis, CoreVector3& zAxis, CoreVector3& pos)
{
	xAxis = this->xAxis;
	yAxis = this->yAxis;
	zAxis = this->zAxis;
	pos = this->pos;
}

void CoreCamera::GetView(CoreVector3 &pos, CoreVector3& lookAt, CoreVector3 &up)
{
	pos =this->pos;
	lookAt = this->zAxis + pos;
	up = this->up;
}

void CoreCamera::generateViewFromPrivateVariables()
{
	view = CoreMatrix4x4Translation(-pos) * CoreMatrix4x4(xAxis.x, yAxis.x, zAxis.x, 0.0f,
														  xAxis.y, yAxis.y, zAxis.y, 0.0f,
														  xAxis.z, yAxis.z, zAxis.z, 0.0f,
														  0.0f,     0.0f,     0.0f,  1.0f);
	calcFrustumData();
}

// MouseLook function
void CoreCamera::MouseLook(float deltaX, float deltaY)
{
	// Screen rotation X = rot will always be handled as rotation around the up vector
	CoreMatrix4x4 rot = CoreMatrix4x4RotationAxis(xAxis, deltaY) * CoreMatrix4x4RotationAxis(up, deltaX);
	xAxis.TransformNormalThis(rot);
	yAxis.TransformNormalThis(rot);
	zAxis.TransformNormalThis(rot);

	generateViewFromPrivateVariables();
}

// Sets an effect variable to the world*view*projection matrix
void CoreCamera::WorldViewProjectionToEffectVariable(ID3DX11EffectMatrixVariable* viewVar, CoreMatrix4x4 &world)
{
	viewVar->SetMatrix((world * view * projection).arr);
}

// Sets an effect variable to the world*view matrix
void CoreCamera::WorldViewToEffectVariable(ID3DX11EffectMatrixVariable* viewVar, CoreMatrix4x4 &world)
{
	viewVar->SetMatrix((world * view).arr);
}

CoreMatrix4x4 CoreCamera::GetClipProjectionMatrix(CorePlane &ClipPlane)
{
	CorePlane vProjClipPlane;
	CoreMatrix4x4 mClipProj;
	CoreMatrix4x4 mViewProj = view * projection;
	ClipPlane.NormalizeThis();
	mViewProj.InvertThis();
	mViewProj.TransposeThis();	
	// die Clipping-Ebene in den Clip-Space transformieren
	vProjClipPlane = CorePlaneTransform(ClipPlane, mViewProj);
	if (vProjClipPlane.d == 0)
	{
		// die Ebene ist senkrecht zur Near-Clipping-Plane
		return projection;
	} else if (vProjClipPlane.d > 0)
	{
		// Ebene umkehren
		ClipPlane = CorePlane(-ClipPlane.a, -ClipPlane.b,
							  -ClipPlane.c, -ClipPlane.d);
		// Clipping-Ebene in den Clip-Space transformieren
		vProjClipPlane = CorePlaneTransform(ClipPlane, mViewProj);
	}
	mClipProj = CoreMatrix4x4();	//Identity

	// Daten in die z-Spalte schreiben
	mClipProj._13 = vProjClipPlane.a;
	mClipProj._23 = vProjClipPlane.b;
	mClipProj._33 = vProjClipPlane.c;
	mClipProj._43 = vProjClipPlane.d;
	return projection * mClipProj;
}

// Set parallel projection
void CoreCamera::SetProjectionParallel(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
	// TODO: Frustum
	projection = CoreMatrix4x4(2.0f / (xmax - xmin),			0.0f,							0.0f,					0.0f,
							   0.0f,							2.0f / (ymax - ymin),			0.0f,					0.0f,
							   0.0f,							0.0f,							1.0f / (zmax - zmin),   0.0f,
							   (xmin + xmax) / (xmin - xmax),   (ymin + ymax) / (ymin - ymax),  zmin / (zmin - zmax),   1.0f);
}

void CoreCamera::calcFrustumData()
{
	CoreVector3 dir, nc, fc;
	CoreVector3 ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr;

	nc = pos + zAxis * m_nearPlane;
	fc = pos + zAxis * m_farPlane;

	ntl = nc + yAxis * nh - xAxis * nw;
	ntr = nc + yAxis * nh + xAxis * nw;
	nbl = nc - yAxis * nh - xAxis * nw;
	nbr = nc - yAxis * nh + xAxis * nw;

	ftl = fc + yAxis * fh - xAxis * fw;
	ftr = fc + yAxis * fh + xAxis * fw;
	fbl = fc - yAxis * fh - xAxis * fw;
	fbr = fc - yAxis * fh + xAxis * fw;

	frustumPlanes[TOP] = CorePlane(ntr, ntl, ftl);
	frustumPlanes[BOTTOM] = CorePlane(nbl, nbr, fbr);
	frustumPlanes[LEFT] = CorePlane(ntl, nbl, fbl);
	frustumPlanes[RIGHT] = CorePlane(nbr, ntr, fbr);
	frustumPlanes[NEARP] = CorePlane(ntl, ntr, nbr);
	frustumPlanes[FARP] = CorePlane(ftr, ftl, fbl);
}