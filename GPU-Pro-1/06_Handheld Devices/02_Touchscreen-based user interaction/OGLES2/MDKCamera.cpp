/******************************************************************************

 @File         MDKCamera.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2009 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  Multiple classes for multiple camera control

******************************************************************************/

#include "MDKCamera.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "MDKMath.h"

namespace MDK {
	namespace GLTools {

		/*********************************************************
		 * Camera 
		 *
		 *********************************************************/
		Camera::Camera()
		{
			mWorld = PVRTMat4::Identity();
		}

		PVRTMat3 Camera::GetModelViewITMatrix()
		{
			return GetModelViewITMatrix(mView * mWorld);
		}
		PVRTMat3 Camera::GetModelViewITMatrix(PVRTMat4 mModelView)
		{
			PVRTMat4 mModelViewI, mModelViewIT;
			mModelViewI  = mModelView.inverse();
			mModelViewIT = mModelViewI.transpose();
			return PVRTMat3(mModelViewIT);
		}

		PVRTMat4 Camera::LookAt(const PVRTVec3& vEye, const PVRTVec3& vAt, const PVRTVec3& vUp)
		{
			PVRTVec3 vForward, vUpNorm, vSide;
			PVRTMat4 result;

			vForward = vEye - vAt;

			vForward.normalize();
			vUpNorm = vUp.normalized();
			vSide   = vUpNorm.cross(vForward).normalized();		
			vUpNorm = vForward.cross(vSide);

			result.f[0]=vSide.x;	result.f[4]=vSide.y;	result.f[8]=vSide.z;		result.f[12]=0;
			result.f[1]=vUpNorm.x;	result.f[5]=vUpNorm.y;	result.f[9]=vUpNorm.z;		result.f[13]=0;
			result.f[2]=vForward.x; result.f[6]=vForward.y;	result.f[10]=vForward.z;	result.f[14]=0;
			result.f[3]=0;			result.f[7]=0;			result.f[11]=0;				result.f[15]=f2vt(1);

			result.postTranslate(-vEye);
			return result;
		}
		// NOTE: does not work if vForward is multiple of (0,1,0)
		PVRTVec3 Camera::UpVector(const PVRTVec3& vEye, const PVRTVec3& vAt)
		{
			PVRTVec3 vForward = (vEye - vAt).normalize();

			PVRTVec3 vUp = PVRTVec3(0.0, 1.0, 0.0);

			PVRTVec3 vSide = vUp.cross(vForward).normalized();		
			return vForward.cross(vSide);
		}

		/*********************************************************
		 * YawPitchCamera provides pitch and yaw angles management 
		 * for automatic creation of the View Matrix
		 *
		 *********************************************************/

		YawPitchCamera::YawPitchCamera() {

			fPitch = 0.0f;
			fYaw = 0.0f;


			// constraints
			bPitchFree = true;
			fMinPitch = -PVRT_PI;
			fMaxPitch = PVRT_PI;

			bYawFree = true;
			fMinYaw = -PVRT_PI;
			fMaxYaw = PVRT_PI;

		}


		void YawPitchCamera::UpdatePitch(float dp)
		{
			fPitch += dp;
			if (!bPitchFree)
			{
				if (fPitch > fMaxPitch)
				{
					fPitch = fMaxPitch;
				}
				if (fPitch < fMinPitch)
				{
					fPitch = fMinPitch;
				}
			}
		}

		void YawPitchCamera::UpdateYaw(float dy)
		{
			fYaw += dy;
			if (!bYawFree)
			{
				if (fYaw > fMaxYaw)
				{
					fYaw = fMaxYaw;
				}
				if (fYaw < fMinYaw)
				{
					fYaw = fMinYaw;
				}
			}
		}
		/*********************************************************
		 * SphereCamera 
		 *
		 *********************************************************/
		SphereCamera::SphereCamera()
		{
			bRadiusFree = true;
			fMinRadius = 0.1f;
			fMaxRadius = 100.0f;
			fRadius = 1.0f;
		}

		void SphereCamera::UpdateRadius(float mult)
		{
			fRadius *= mult;
			if(!bRadiusFree)
			{
				if (fRadius > fMaxRadius)
				{
					fRadius = fMaxRadius;
				}	
				if (fRadius < fMinRadius)
				{
					fRadius = fMinRadius;
				}
			}	
		}

		void SphereCamera::UpdateViewMatrix()
		{
			PVRTMat3 mCamera = PVRTMat3::RotationY(fYaw) * PVRTMat3::RotationX(fPitch);
			vFrom = mCamera * PVRTVec3(0.0, 0.0, fRadius);
			vTo = PVRTVec3(0.0, 0.0, 0.0);

			/*
			 * If the target direction is always the origin, the up vector can be obtained just by
			 * multiplying (0,1,0) by the relative orientation (mCamera) given by the (yaw, pitch) angles.
			 */
			vUp = mCamera * PVRTVec3(0.0, 1.0, 0.0);
			mView = PVRTMat4::LookAtRH(vFrom, vTo, vUp);

			/*
			 * If the target direction is arbitrary, the up vector cannot be calculated easily as in the previous case.
			 * Camera::LookAt provides a modification of the PVRTMat4::LookAtRH function, giving an orthogonal viewing
			 * matrix even if the direction and the up vector are not.
			 * If a class with arbirtrary camera position and target is implemented, the following two lines of code need to
			 * be used to calculate the view matrix
			 */
			// PVRTVec3 vUp = PVRTVec3(0.0, 1.0, 0.0);
			// mView = LookAt(vFrom, vTo, vUp);
		}

		/*********************************************************
		 * TargetCamera 
		 *********************************************************/
		TargetCamera::TargetCamera()
		{
			vTo = PVRTVec3(0.0f);
			fBaseYaw = 0.0f;
			fBasePitch = 0.0f;
		}

		void TargetCamera::Set(const PVRTVec3 &from, const PVRTVec3 &to)
		{
			vFrom = from;
			vTo = to;
			vUp = UpVector(vFrom, vTo);

			PVRTVec3 vDir = vTo - from;

			fRadius = vDir.length();

			fYaw = PVRT_PI - atan2(vDir.x, vDir.z);
			fPitch = -atan2(vDir.y, sqrt(vDir.z * vDir.z + vDir.x * vDir.x));

			fBaseYaw = 0.0f;
			fBasePitch = 0.0f;
		}
		void TargetCamera::SetTo(PVRTVec3 &to)
		{
			vTo = to;
		}

		void TargetCamera::UpdateTarget(PVRTVec3 &to)
		{
			// Get the from vector from the old orientation
			PVRTMat3 mCamera = PVRTMat3::RotationY(fYaw/* + fBaseYaw*/) * PVRTMat3::RotationX(fPitch/* + fBasePitch*/);
			vFrom = vTo + mCamera * PVRTVec3(0.0, 0.0, fRadius);

			// Update the target
			vTo = to;

			// Recalculate angles and radius
			vUp = UpVector(vFrom, vTo);

			PVRTVec3 vDir = vTo - vFrom;

			fRadius = vDir.length();

			fYaw = PVRT_PI - atan2(vDir.x, vDir.z);
			fPitch = -atan2(vDir.y, sqrt(vDir.z * vDir.z + vDir.x * vDir.x));

			mView = LookAt(vFrom, vTo, vUp);
		}



		void TargetCamera::UpdateViewMatrix()
		{
			float yaw = fYaw + fBaseYaw;
			if (!bYawFree)
			{
				if (yaw > fMaxYaw)
				{
					fBaseYaw = fMaxYaw - fYaw;
				}
				if (yaw < fMinYaw)
				{
					fBaseYaw = fMinYaw - fYaw;
				}
			}

			float pitch = fPitch + fBasePitch;
			if (!bPitchFree)
			{
				if (pitch > fMaxPitch)
				{
					fBasePitch = fMaxPitch - fPitch;
				}
				if (pitch < fMinPitch)
				{
					fBasePitch = fMinPitch - fPitch;
				}
			}

			// Calculate From and Up vectors given the yaw, pitch and radius values
			PVRTMat3 mCamera = PVRTMat3::RotationY(fYaw + fBaseYaw) * PVRTMat3::RotationX(fPitch + fBasePitch);
			vFrom = vTo + mCamera * PVRTVec3(0.0, 0.0, fRadius);

			vUp = UpVector(vFrom, vTo);
			// Finally calculate the view matrix
			mView = LookAt(vFrom, vTo, vUp);
		}
	}
}
