

#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include <Common/Common.hpp>
#include <Common/Incopiable.hpp>
#include <Math\Matrix\Matrix.hpp>

///<
class Camera : public Incopiable
{
	enum State
	{
		ORBIT=0,
		FREE=1,
		N
	};

	///< Orbit:
	Vector3f m_x;
	Vector3f m_up;
	Vector3f m_at;
	Vector3f m_axisOrbit;

	State				m_State;	

	Vector2ui			m_wh;

	Matrix4f			m_View;
	Matrix4f			m_Proj;

	static Camera		m_instance;

	void UpdateOrbit					();
	void UpdateFree						();

public:

	///<
	explicit Camera();	
	~Camera();
	
	void Create				(const uint32 _w, const uint32 _h);
	void Update				();

	static Camera&			Get						()	{return m_instance;}
	
	const Matrix4f			LightView() const;
	const Matrix4f&			View					() const;
	const Matrix4f&			Projection				() const;

	///< Orbit:
	void SetOrbitAxis(const Vector3f _x){m_axisOrbit=_x;}
	void SetX(const Vector3f _x){m_x=_x;}
	void SetUp(const Vector3f _up){m_up=_up;}
	void SetAt(const Vector3f _at){m_at=_at;}
	void SetDefaultOrbit();

	Vector2ui				ScreenDimensions		() const {return m_wh;}
	Vector3f				X						() const;
	Vector3f				Dir						() const;
	Vector3f				Up						() const;
	Vector3f				Right					() const;

};


#endif