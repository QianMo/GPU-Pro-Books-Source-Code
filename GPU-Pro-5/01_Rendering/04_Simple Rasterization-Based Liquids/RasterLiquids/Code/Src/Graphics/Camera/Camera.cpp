

#include <Graphics/Camera/Camera.hpp>

#include <Input/Keyboard.hpp>

#include <Math/Math.hpp>
#include <Math/Vector/Vector.hpp>
#include <Math/Quaternion/Quaternion.hpp>

#include <..\\External\\AntTweakBar\\include\\AntTweakBar.h>

///<
Camera	Camera::m_instance;

void Camera::SetDefaultOrbit()
{
	m_x=Vector3f(0, 24.0f, 72.0f);
	m_up=Vector3f(0,1,0);
	m_at=Vector3f(0,0,0);
	m_axisOrbit=M::yAxis;
}

Camera::Camera() : m_State(ORBIT)
{
	///< 
	SetDefaultOrbit();
}

Camera::~Camera()
{ 

}

///<
void Camera::Create(const uint32 _w, const uint32 _h)
{
	m_wh = Vector2ui(_w,_h);
	Update();
}


///<
const Matrix4f& Camera::View() const
{
	return m_View;
}

///<
const Matrix4f& Camera::Projection() const
{
	return m_Proj;
}

 Vector3f Camera::X() const
{
	return Vector3f(m_View(0,3),m_View(1,3),m_View(2,3));
}

///<
Vector3f Camera::Right()const
{
	return Vector3f(m_View(0,0),m_View(1,0),m_View(2,0));
}

///<
Vector3f Camera::Up()const
{
	return Vector3f(m_View(0,1),m_View(1,1),m_View(2,1));
}

///<
Vector3f Camera::Dir()const
{
	return Vector3f(m_View(0,2),m_View(1,2),m_View(2,2));
}


Matrix4f LookAt(Vector3f _x, Vector3f _at, Vector3f _up)
{
	Vector3f z = -(_x-_at).Normalize();
	Vector3f x = CrossProduct(_up, z).Normalize();
	Vector3f y = CrossProduct(z, x);

	Vector3f X;
	X[0]=-DotProduct(x,_x);
	X[1]=-DotProduct(y,_x);
	X[2]=-DotProduct(z,_x);

	Matrix4f view=Matrix4f::Identity();
	for(uint32 i=0; i<3; ++i)
	{
		view(3,i)=X[i];

		view(i,0)=x[i];
		view(i,1)=y[i];
		view(i,2)=z[i];
	}
	
	return view.Transpose();
}

///<
Matrix4f PerspectiveLH(const float32 _w, const float32 _h, float32 _zNear, float32 _zFar)
{
	float32 h = 1.0f/tan( M::Pi/8.0f);
	float32 aspect = M::Divide<float32>(_w, _h);

	Matrix4f proj;
	proj(0,0)= h/aspect;
	proj(1,1)= h;
	proj(2,2)= -_zFar/(_zNear-_zFar); 
	proj(2,3)= 1.0f;
	proj(3,2)=_zNear*_zFar/(_zNear-_zFar);

	return proj;
}

///<
const Matrix4f Camera::LightView() const
{
	return LookAt(Vector3f(2.0f, 35.0f, 8.0f),Vector3f(0,1,0),Vector3f(0,1,0));
}

/////////////////////////////////////////////////////////

void Camera::UpdateOrbit()
{
	Vector3f At = m_at;
	Vector3f Up = m_up;

	Keyboard& keyboard = Keyboard::Get();

	float32 dx=0.04f;
	Vector2f dh(1,0);
	if (keyboard.Key(DIK_W))
		dh[0]-=dx;
	if (keyboard.Key(DIK_S))
		dh[0]+=dx;
	if (keyboard.Key(DIK_A))
		dh[1]+=dx;
	if (keyboard.Key(DIK_D))
		dh[1]-=dx;
	
	m_x = dh.x()*m_x;
	m_x = Quaternionf::GenRotation(dh.y(), m_axisOrbit).Rotate(m_x);
	
	TwBar* pBar = TwGetBarByIndex(1);
	if (pBar)
	{	
		Vector2i iMenuSize;		
		TwGetParam(pBar, NULL, "size", TW_PARAM_INT32, 2, iMenuSize.Begin());
		if (keyboard.MouseX().x() < 1.3f*iMenuSize.x())
			dx=0;
	}

	if (!keyboard.MouseClick1())
		dx=0;
	
	m_View = LookAt(m_x,At,Up);
	m_Proj = PerspectiveLH(M::SCast<float32>(m_wh.x()), M::SCast<float32>(m_wh.y()), 1.1f, 1000.0f);
}

void Camera::Update()
{
	UpdateOrbit();
}