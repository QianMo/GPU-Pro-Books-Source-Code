/******************************************************************************

 @File         MDKCamera.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2009 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  	Different camera classes with multiple functionality are provided.
		See the comments on each class for details
******************************************************************************/


#ifndef _MDK_CAMERA_H_
#define _MDK_CAMERA_H_

#include "../PVRTMatrix.h"
#include "../PVRTVector.h"


namespace MDK {
	namespace GLTools {

		//! Camera base class
		/*!
			Camera base class exposing accessors and modifiers for world, view and projection matrices.
		*/
		class Camera
		{
		protected:
			PVRTMat4 mProj;
			PVRTMat4 mView;
			PVRTMat4 mWorld;

			PVRTVec3 vFrom;
			PVRTVec3 vTo;
			PVRTVec3 vUp;
		public:
			//! Constructor
			/*!

			*/
			Camera();

			//! Destructor
			/*!
		
			*/
			virtual ~Camera() { }

			// Projection matrix

			//! Sets the projection matrix.
			/*!
				\param proj The new projection matrix.	
			*/
			void SetProjectionMatrix(const PVRTMat4 &proj) { mProj = proj; }

			//! Gets the projection matrix.
			/*!
				\return A copy of the camera's projection matrix.
			*/
			PVRTMat4 GetProjectionMatrix() { return mProj; }

			//! Gets a pointer to the projection matrix.
			/*!
				\return A pointer to the first element in the camera's projection matrix.
			*/
			float *GetProjectionMatrixPtr() { return mProj.f; }
			
			//! Sets the view matrix.
			/*!
				\param view The new view matrix.
			*/
			void SetViewMatrix(const PVRTMat4 &view) { mView = view; }

			//! Gets the view matrix.
			/*!
				\return A copy of the camera's view matrix.
			*/
			PVRTMat4 GetViewMatrix() { return mView; }

			//! Gets a pointer to the view matrix.
			/*!
				\return A pointer to the first element in the camera's view matrix.
			*/
			float *GetViewMatrixPtr() { return mView.f; }

			//! Sets the world matrix.
			/*!
				\param world The new world matrix.
			*/
			void SetWorldMatrix(const PVRTMat4 &world) { mWorld = world; }
			
			//! Gets the view-projection matrix.
			/*!
				\return The product of the camera's projection and view matrices.
			*/
			PVRTMat4 GetViewProjectionMatrix() { return mProj * mView; }
			
			//! Gets the world-view-projection matrix.
			/*!
				\return The product of the camera's projection, view and world matrices.
			*/
			PVRTMat4 GetWorldViewProjectionMatrix() { return mProj * mView * mWorld; }
			
			//! Gets world-view-projection matrix, passing the world matrix as a parameter.
			/*!
				\param world Reference to the world matrix.
				\return The product of the camera's projection and view matrices, multiplied by the specified world matrix.
			*/
			PVRTMat4 GetWorldViewProjectionMatrix(const PVRTMat4 &world) { return mProj * mView * world; }

			//! Returns the world-view matrix, specifying the world matrix as a parameter.
			/*!
				\param world Reference to the world matrix.
				\return The product of the camera's view matrix and the specified world matrix.
			*/
			PVRTMat4 GetWorldViewMatrix(const PVRTMat4 &world) { return mView * world; }

			//! Gets the world-view matrix.
			/*!
				\return The product of the camera's view and world matrices.
			*/
			PVRTMat4 GetWorldViewMatrix() { return mView * mWorld; }

			//! Gets the inverse transpose of the model-view matrix.
			/*!
				\return The inverse transpose of the product of the camera's model and view matrices.
			*/
			PVRTMat3 GetModelViewITMatrix();
			 
			// FIXME: Speed up these functions to store some of these values:

			//! Gets the camera's position.
			/*!
				Returns the camera's position, or the 'eye' vector.
				\return The position of the camera.
			*/
			virtual PVRTVec3 GetPosition(){ return vFrom;}

			//! Gets the camera's 'up' vector.
			/*!
				Returns the vector orthogonal to the camera's 'to' direction and 'right' directions.
				\return The camera's up vector.
			*/
			virtual PVRTVec3 GetUp(){ return vUp;}

			//! Gets the camera's 'to' vector.
			/*!	
				Returns the camera's target position in world space. 
				\return The position of the camera's target.
			*/
			virtual PVRTVec3 GetTo(){ return vTo;}

			//! Gets the direction that the camera is pointing in.
			/*!
				Returns the direction the camera is facing, obtained using normalize(to-from).
				\return The camera's direction.
			*/
			virtual PVRTVec3 GetDirection(){ return (vTo - vFrom).normalize();}

			//! Gets the camera's 'right' vector.
			/*!
				Returns the vector orthogonal to the camera's 'to' and 'up' directions.
				\return The camera's right vector.
			*/
			virtual PVRTVec3 GetRight(){ return (vUp.cross(GetDirection())).normalize();}

			//! Calculates the inverse transpose of the specified matrix.
			/*!
				Returns the 3x3 inverse transpose of the specified 4x4 matrix. This is useful for lighting and normals.
				\param mModelView The inverse transpose of the specified matrix.
			*/
			static PVRTMat3 GetModelViewITMatrix(PVRTMat4 mModelView);

			//! Calculates a view matrix based on eye, target and up values.
			/*!
				Calculates a view matrix based on an eye position, target position, and 'up' direction which should be orthogonal to the camera's view direction.
				\param vEye Reference to the eye position.
				\param vAt Reference to the target position.
				\param vUp Reference to the up direction.
				\return The computed view matrix.
			*/
			static PVRTMat4 LookAt(const PVRTVec3& vEye, const PVRTVec3& vAt, const PVRTVec3& vUp);

			//! Calculates an up direction based on eye and target position.
			/*!
				Calculates an up direction corresponding to the direction obtained by normalize(vAt-vEye). This is done by calculating the cross product of the direction with (0, 1, 0) to obtain a 'right' direction, then taking the cross product of this right direction with the view direction. The result is then normalized.
				\param vEye Reference to the eye position.
				\param vAt Reference to the target position.
				\return 
			*/
			static PVRTVec3 UpVector(const PVRTVec3& vEye, const PVRTVec3& vAt);
		};

		//! Camera controlled by yaw and pitch angles.
		/*!
	 		YawPitchCamera provides pitch and yaw angles management for automatic creation of the View Matrix. This class cannot be instantiated, use one of its derived versions instead.
			Note - Yaw and Pitch values can be updated; after that UpdateViewMatrix needs to be called for the new angles to be applied.
		*/
		class YawPitchCamera : public Camera
		{
		protected:
			float fPitch;
			float fYaw;

			bool bPitchFree;
			float fMinPitch;
			float fMaxPitch;

			bool bYawFree;
			float fMinYaw;
			float fMaxYaw;

		public:
			//! Constructor
			YawPitchCamera();

			//! Destructor
			virtual ~YawPitchCamera() { }

			//! Sets the camera's pitch.
			/*!
				\param pitch New pitch value, in radians.
			*/
			void SetPitch(float pitch) { fPitch = pitch; }

			//! Updates the camera's pitch relative to its current value.	
			/*!
				\param dp Value to be added to camera's current pitch, in radians.
			*/
			void UpdatePitch(float dp);

			//! Gets the camera's pitch.
			/*!
				\return Camera's current pitch, in radians.
			*/
			float GetPitch() { return fPitch; }

			//! Sets constraints for the camera's pitch.
			/*!
				\param min The minimum value the pitch can become, in radians.
				\param max The maximum value the pitch can become, in radians.
			*/
			void SetPitchConstraints(float min, float max) { bPitchFree = false; fMinPitch = min; fMaxPitch = max; }

			//! Unconstrains the camera's pitch.
			void UnsetPitchConstraints() { bPitchFree = true; }

			//! Sets the camera's yaw.
			/*!
				\param yaw New yaw value, in radians.
			*/
			void SetYaw(float yaw) { fYaw = yaw; }

			//! Updates the camera's yaw relative to its current value.
			/*!
				\param dy Value to be added to camera's current yaw, in radians.
			*/
			void UpdateYaw(float dy);

			//! Gets the camera's yaw.
			/*!
				\return Camera's current yaw, in radians.
			*/ 
			float GetYaw() { return fYaw; }

			//! Sets constraints for the camera's yaw.
			/*!
				\param min The minimum value the yaw can become, in radians.
				\param max The maximum value the yaw can become, in radians.
			*/
			void SetYawConstraints(float min, float max) { bYawFree = false; fMinYaw = min; fMaxYaw = max; }

			//! Unconstrains the camera's yaw.
			void UnsetYawConstraints() { bYawFree = true; }

			//! Returns a string containing the camera's current yaw and pitch.
			char *ToString(char *buffer) { sprintf(buffer, "yaw = %.2f, pitch = %.2f", fYaw, fPitch); return buffer; }

			virtual void UpdateViewMatrix() = 0;
		};

		//! Camera controlled by yaw, pitch and radius.
		/*!
			SphereCamera adds a radius to YawPitchCamera the resulting camera has target on the origin and position in a spherical coordinate system defined by (yaw, pitch, radius)
			This camera is useful for spinning around objects and can be integrated with the touchscreen module to associate yaw and pitch angles to the motion in the two directions.
		*/
		class SphereCamera : public YawPitchCamera
		{
		protected:
			float fRadius;
			bool bRadiusFree;
			float fMinRadius;
			float fMaxRadius;

		public:
			//! Constructor
			SphereCamera();

			//! Destructor
			virtual ~SphereCamera() { }

			//! Sets constraints for the camera's radius (distance from the origin to the camera).
			/*!
				\param min The minimum distance from the origin.
				\param max The maximum distance from the origin.
			*/
			void SetRadiusConstraints(float min, float max) { bRadiusFree = false; fMinRadius = min; fMaxRadius = max; }

			//! Unconstrains the camera's radius (distance from the origin to the camera).
			void UnsetRadiusConstraints() { bRadiusFree = true; }

			//! Sets the camera's radius (distance from the origin to the camera).
			/*!
				\param radius The new distance from eye to target.
			*/
			virtual void SetRadius(float radius) { fRadius = radius; }

			//! Gets the camera's radius (distance from the origin to the camera).
			/*!
				\return The distance from the origin to the camera.
			*/
			float GetRadius() { return fRadius; }

			//! Multiplies the current radius by a given value.
			/*!
				\param mult The value that the current distance from camera to origin will be multiplied by.
			*/	
			void UpdateRadius(float mult);

			//! Updates the camera's view matrix.
			/*!
				This always needs to be called after yaw, pitch or radius have been modified.
			*/
			virtual void UpdateViewMatrix();
		};


		//! Camera based on SphereCamera but with a variable target position.
		/*!
	 		TargetCamera is a generalization of SphereCamera, where the target position can be arbitrary and not fixed on the origin.
			Before updating the view matrix, the function Set has to be called as it converts the (from, to) values to ((yaw, pitch, radius), to). Once this is done, further  adjustments can be done with the fYaw and fPitch values in YawPitchCamera.
			NOTE: When calling CPVRTModelPOD::GetCameraPos, the returned vTo vector is such that (vTo - vFrom).length() == 1. This is not the case for the TargetCamera as the distance between "position" and "to" is always the correct one.
		*/
		class TargetCamera : public SphereCamera
		{
			float fBaseYaw;
			float fBasePitch;
		public:
			//! Constructor
			TargetCamera();

			//! Destructor
			virtual ~TargetCamera() { }

			//! Sets the camera's yaw, pitch and radius from 'from' and 'to' positions.
			/*!
				This is an initialization method that sets the target to "to", and calculates yaw, pitch and radius values so that the position will match the passed parameter "from"
				In other words, set finds the yaw, pitch and radius values that correspond to (from - to) in a spherical coordinate system.
				\param from Reference to the new eye position.
				\param to Reference to the new target position.
			*/
			void Set(const PVRTVec3 &from, const PVRTVec3 &to);

			//! Updates the target, without altering the yaw, pitch and radius.
			/*!
				\param to Reference to the new target position.
			*/
			void SetTo(PVRTVec3 &to);

			//! Updates the target position, adjusting yaw, pitch and radius accordingly.
			/*!
				UpdateTarget changes the target and updates yaw, pitch and radius accordingly, without altering the previously saved position.
				\param to Reference to the new target position.
			*/
			void UpdateTarget(PVRTVec3 &to);
			
			//! FIXME
			/*!
				\return
			*/
			float GetBaseYaw() { return fBaseYaw; }

			//! FIXME
			/*!
				\param baseYaw
			*/
			void SetBaseYaw(float baseYaw) { fBaseYaw = baseYaw; }

			//! FIXME
			/*!
				\return
			*/
			float GetBasePitch() { return fBasePitch; }

			//! FIXME
			/*!
				\param basePitch
			*/
			void SetBasePitch(float basePitch) { fBasePitch = basePitch; }

			//! Updates the view matrix.
			/*!
				This always needs to be called after yaw, pitch or radius have been updated. See Flowers demo for extensive TargetCamera usage.
			*/
			virtual void UpdateViewMatrix();
		};



	}
}
#endif
