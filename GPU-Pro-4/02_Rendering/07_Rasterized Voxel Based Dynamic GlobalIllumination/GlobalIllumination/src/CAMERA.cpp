#include <stdafx.h>
#include <DEMO.h>
#include <CAMERA.h>

bool CAMERA::Init(float fovy,float nearClipDistance,float farClipDistance)
{
	this->fovy = fovy;
	this->nearClipDistance = nearClipDistance;
	if(nearClipDistance==0.0f)
		return false;
	this->farClipDistance = farClipDistance;
	nearFarClipDistance = farClipDistance-nearClipDistance;
	aspectRatio = (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT;
	halfFarHeight = tan(fovy*M_PI/360)*farClipDistance;
	halfFarWidth =  halfFarHeight*aspectRatio;
	projMatrix.SetPerspective(fovy,aspectRatio,nearClipDistance,farClipDistance);	
	invProjMatrix = projMatrix.GetInverse();	

	UNIFORM_LIST uniformList;
	uniformList.AddElement("viewMatrix",MAT4_DT);
	uniformList.AddElement("invTransposeViewMatrix",MAT4_DT);
	uniformList.AddElement("projMatrix",MAT4_DT);
	uniformList.AddElement("viewProjMatrix",MAT4_DT);
	uniformList.AddElement("frustumRays",VEC4_DT,4);
	uniformList.AddElement("position",VEC3_DT);
	uniformList.AddElement("nearClipDistance",FLOAT_DT);
	uniformList.AddElement("farClipDistance",FLOAT_DT);
	uniformList.AddElement("nearFarClipDistance",FLOAT_DT);
	uniformBuffer = DEMO::renderer->CreateUniformBuffer(CAMERA_UB_BP,uniformList);
	if(!uniformBuffer)
		return false;
	
	UpdateUniformBuffer();
	
	return true;
}

void CAMERA::UpdateUniformBuffer()
{
	float *uniformBufferData = viewMatrix;
	uniformBuffer->Update(uniformBufferData);
}

void CAMERA::Update(const VECTOR3D &position,const VECTOR3D &rotation)
{
	this->position = position;
	this->rotation = rotation;
	MATRIX4X4 xRotMatrix,yRotMatrix,zRotMatrix,transMatrix,rotMatrix;
	xRotMatrix.SetRotation(VECTOR3D(0.0f,1.0f,0.0f),-rotation.x);
	yRotMatrix.SetRotation(VECTOR3D(1.0f,0.0f,0.0f),rotation.y);
	zRotMatrix.SetRotation(VECTOR3D(0.0f,0.0f,1.0f),rotation.z);
	transMatrix.SetTranslation(-position);
	rotMatrix = zRotMatrix*yRotMatrix*xRotMatrix;
	viewMatrix = rotMatrix*transMatrix;
	viewProjMatrix = projMatrix*viewMatrix;
	invTransposeViewMatrix = viewMatrix.GetInverseTranspose();

	direction.Set(-viewMatrix.entries[2],-viewMatrix.entries[6],-viewMatrix.entries[10]);
  direction.Normalize();
	VECTOR3D up(viewMatrix.entries[1],viewMatrix.entries[5],viewMatrix.entries[9]);
	up.Normalize();
	VECTOR3D right(viewMatrix.entries[0],viewMatrix.entries[4],viewMatrix.entries[8]);
	right.Normalize();

	VECTOR3D farDir = direction*farClipDistance;
	VECTOR3D tmp;

	// left/ lower corner
	tmp = farDir-(up*halfFarHeight)-(right*halfFarWidth);
	frustumRays[0].Set(tmp);

	// right/ lower corner
	tmp = farDir-(up*halfFarHeight)+(right*halfFarWidth);
	frustumRays[1].Set(tmp);

	// left/ upper corner
	tmp = farDir+(up*halfFarHeight)-(right*halfFarWidth);
	frustumRays[2].Set(tmp);
	
	// right/ upper corner
	tmp = farDir+(up*halfFarHeight)+(right*halfFarWidth);
	frustumRays[3].Set(tmp);

	UpdateUniformBuffer();
}





