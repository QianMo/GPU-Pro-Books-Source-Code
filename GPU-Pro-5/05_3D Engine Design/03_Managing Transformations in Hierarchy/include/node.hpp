#ifndef NODE_HPP
#define NODE_HPP

#include <cstring>
#include <vector>

#include "math/quaternion.hpp"



using namespace std;



class Node
{
public:
	Node();

	void SetParent(Node* newParent);
	void SetChild(Node* newChild) { newChild->SetParent(this); }

	// local translation sets
	void SetLocalTranslation(float x, float y, float z, bool useLocalRotation = false);
	void SetLocalTranslation(const Vector3& translation, bool useLocalRotation = false);
	void LocalTranslate(float x, float y, float z, bool useLocalRotation = false);
	void LocalTranslate(const Vector3& translation, bool useLocalRotation = false);

	// global translation sets
	void SetGlobalTranslation(float x, float y, float z);
	void SetGlobalTranslation(const Vector3& translation);
	void GlobalTranslate(float x, float y, float z);
	void GlobalTranslate(const Vector3& translation);

	// local rotation sets
	void SetLocalRotation(const Quaternion& rotation);
	void LocalRotate(const Quaternion& rotation);
	void SetLocalEulerAngles(float x, float y, float z);
	void SetLocalEulerAngles(const Vector3& eulerAngles);

	// global rotation sets
	void SetGlobalRotation(const Quaternion& rotation);
	void GlobalRotate(const Quaternion& rotation);
	void SetGlobalEulerAngles(float x, float y, float z);
	void SetGlobalEulerAngles(const Vector3& eulerAngles);

	// local scale sets
	void SetLocalScale(float x, float y, float z);
	void SetLocalScale(const Vector3& scale);

	// local gets

	const Vector3& GetLocalTranslation() const;
	const Quaternion& GetLocalRotation() const;
	Vector3 GetLocalEulerAngles() const;
	const Vector3& GetLocalScale() const;

	Vector3 GetLocalRight();
	Vector3 GetLocalUp();
	Vector3 GetLocalBackward();
	Vector3 GetLocalForward();

	const Matrix& GetLocalMatrix() const;
	Matrix GetLocalMatrix_TranslationAndRotation() const;

	void ComputeLocalBasisVectors(Vector3& right, Vector3& up, Vector3& backward) const;
	void ComputeLocalBasisVectors_Forward(Vector3& right, Vector3& up, Vector3& forward) const;

	// global gets

	Vector3 GetGlobalTranslation() const;
	Quaternion GetGlobalRotation() const;
	Vector3 GetGlobalEulerAngles() const;
	Vector3 GetGlobalLossyScale() const;

	Vector3 GetGlobalRight();
	Vector3 GetGlobalUp();
	Vector3 GetGlobalBackward();
	Vector3 GetGlobalForward();

	Matrix GetGlobalMatrix() const;
	Matrix GetGlobalMatrix_TranslationAndRotation() const;
	Matrix GetGlobalMatrixWithoutLocalMatrix() const;

	void ComputeGlobalBasisVectors(Vector3& right, Vector3& up, Vector3& backward) const;
	void ComputeGlobalBasisVectors_Forward(Vector3& right, Vector3& up, Vector3& forward) const;

	//

	void Debug_Recursive(Node* entity, byte indent = 0)
	{
		for (byte i = 0; i < indent; i++)
			cout << " ";

		cout << entity->name << endl;

		for (uint i = 0; i < entity->children.size(); i++)
			Debug_Recursive(entity->children[i], indent + 2);
	}

public:
	string name;

private:
	void UpdateLocalMatrix();

private:
	Node* parent;
	vector<Node*> children;

	Vector3 localTranslation;
	Quaternion localRotation;
	Vector3 localScale;

	Matrix localMatrix;
};



inline void Node::SetLocalTranslation(float x, float y, float z, bool useLocalRotation)
{
	if (!useLocalRotation)
	{
		localTranslation.x = x;
		localTranslation.y = y;
		localTranslation.z = z;
	}
	else
	{
		Vector3 right, up, backward;
		ComputeLocalBasisVectors(right, up, backward);

		localTranslation = x * right;
		localTranslation += y * up;
		localTranslation += z * backward;
	}

	UpdateLocalMatrix();
}



inline void Node::SetLocalTranslation(const Vector3& translation, bool useLocalRotation)
{
	SetLocalTranslation(translation.x, translation.y, translation.z, useLocalRotation);
}



inline void Node::LocalTranslate(float x, float y, float z, bool useLocalRotation)
{
	if (!useLocalRotation)
	{
		localTranslation.x += x;
		localTranslation.y += y;
		localTranslation.z += z;
	}
	else
	{
		Vector3 right, up, backward;
		ComputeLocalBasisVectors(right, up, backward);

		localTranslation += x * right;
		localTranslation += y * up;
		localTranslation += z * backward;
	}

	UpdateLocalMatrix();
}



inline void Node::LocalTranslate(const Vector3& translation, bool useLocalRotation)
{
	LocalTranslate(translation.x, translation.y, translation.z, useLocalRotation);
}



inline void Node::SetGlobalTranslation(float x, float y, float z)
{
	localTranslation =
		Vector3(Vector4(x, y, z, 1.0f) *
		GetGlobalMatrixWithoutLocalMatrix().GetInversed());

	UpdateLocalMatrix();
}



inline void Node::SetGlobalTranslation(const Vector3& translation)
{
	SetGlobalTranslation(translation.x, translation.y, translation.z);
}



inline void Node::GlobalTranslate(float x, float y, float z)
{
	Vector4 tempLocalTranslation = Vector4(localTranslation);
	Matrix globalMatrixWithoutLocalMatrix = GetGlobalMatrixWithoutLocalMatrix();

	tempLocalTranslation *=
		globalMatrixWithoutLocalMatrix *
		Matrix::Translate(x, y, z) *
		globalMatrixWithoutLocalMatrix.GetInversed();

	localTranslation = Vector3(tempLocalTranslation);

	UpdateLocalMatrix();
}



inline void Node::GlobalTranslate(const Vector3& translation)
{
	GlobalTranslate(translation.x, translation.y, translation.z);
}



inline void Node::SetLocalRotation(const Quaternion& rotation)
{
	localRotation = rotation;

	UpdateLocalMatrix();
}



inline void Node::LocalRotate(const Quaternion& rotation)
{
	localRotation *= rotation;

	UpdateLocalMatrix();
}



inline void Node::SetLocalEulerAngles(float x, float y, float z)
{
	SetLocalRotation(Quaternion::RotateX(x) * Quaternion::RotateY(y) * Quaternion::RotateZ(z));
}



inline void Node::SetLocalEulerAngles(const Vector3& eulerAngles)
{
	SetLocalEulerAngles(eulerAngles.x, eulerAngles.y, eulerAngles.z);
}



inline void Node::SetGlobalRotation(const Quaternion& rotation)
{
	localRotation =
		rotation *
		GetGlobalMatrixWithoutLocalMatrix().GetInversed().GetQuaternion();

	UpdateLocalMatrix();
}



inline void Node::GlobalRotate(const Quaternion& rotation)
{
	Matrix globalMatrixWithoutLocalMatrix = GetGlobalMatrixWithoutLocalMatrix();

	localRotation *=
		globalMatrixWithoutLocalMatrix.GetQuaternion() *
		rotation *
		globalMatrixWithoutLocalMatrix.GetInversed().GetQuaternion();

	UpdateLocalMatrix();
}



inline void Node::SetGlobalEulerAngles(float x, float y, float z)
{
	SetGlobalRotation(Quaternion::RotateX(x) * Quaternion::RotateY(y) * Quaternion::RotateZ(z));
}



inline void Node::SetGlobalEulerAngles(const Vector3& eulerAngles)
{
	SetGlobalEulerAngles(eulerAngles.x, eulerAngles.y, eulerAngles.z);
}



inline void Node::SetLocalScale(float x, float y, float z)
{
	localScale.x = x;
	localScale.y = y;
	localScale.z = z;

	UpdateLocalMatrix();
}

inline void Node::SetLocalScale(const Vector3& scale)
{
	SetLocalScale(scale.x, scale.y, scale.z);
}



inline const Vector3& Node::GetLocalTranslation() const
{
	return localTranslation;
}



inline const Quaternion& Node::GetLocalRotation() const
{
	return localRotation;
}



inline Vector3 Node::GetLocalEulerAngles() const
{
	return GetLocalRotation().ToEulerAngles();
}



inline const Vector3& Node::GetLocalScale() const
{
	return localScale;
}



inline Vector3 Node::GetLocalRight()
{
	return GetLocalRotation().GetRotatedVector(Vector3::AxisX);
}



inline Vector3 Node::GetLocalUp()
{
	return GetLocalRotation().GetRotatedVector(Vector3::AxisY);
}



inline Vector3 Node::GetLocalBackward()
{
	return GetLocalRotation().GetRotatedVector(Vector3::AxisZ);
}



inline Vector3 Node::GetLocalForward()
{
	return -GetLocalBackward();
}



inline const Matrix& Node::GetLocalMatrix() const
{
	return localMatrix;
}



inline Matrix Node::GetLocalMatrix_TranslationAndRotation() const
{
	Matrix temp = GetLocalRotation().GetMatrix();
	memcpy(&temp._[3][0], &GetLocalTranslation(), sizeof(Vector3));
	return temp;
}



inline void Node::ComputeLocalBasisVectors(Vector3& right, Vector3& up, Vector3& backward) const
{
	Quaternion temp = GetLocalRotation();

	right = temp.GetRotatedVector(Vector3::AxisX);
	up = temp.GetRotatedVector(Vector3::AxisY);
	backward = temp.GetRotatedVector(Vector3::AxisZ);
}



inline void Node::ComputeLocalBasisVectors_Forward(Vector3& right, Vector3& up, Vector3& forward) const
{
	ComputeLocalBasisVectors(right, up, forward);
	forward = -forward;
}



inline Vector3 Node::GetGlobalTranslation() const
{
	Matrix globalMatrix = GetGlobalMatrix();
	return Vector3(globalMatrix._[3][0], globalMatrix._[3][1], globalMatrix._[3][2]);
}



inline Quaternion Node::GetGlobalRotation() const
{
	return GetGlobalMatrix_TranslationAndRotation().GetQuaternion();
}



inline Vector3 Node::GetGlobalEulerAngles() const
{
	return GetGlobalRotation().ToEulerAngles();
}



inline Vector3 Node::GetGlobalLossyScale() const
{
	Matrix localMatrix = GetLocalMatrix();
	Matrix localMatrix_TranslationAndRotation = GetLocalMatrix_TranslationAndRotation();

	Node* tempParent;

	tempParent = parent;
	while (tempParent != NULL)
	{
		localMatrix *= tempParent->GetLocalMatrix();
		localMatrix_TranslationAndRotation *= tempParent->GetLocalMatrix_TranslationAndRotation();

		tempParent = tempParent->parent;
	}

	//

	// remove translation parts
	localMatrix._[3][0] = localMatrix._[3][1] = localMatrix._[3][2] = 0.0f;
	localMatrix_TranslationAndRotation._[3][0] = localMatrix_TranslationAndRotation._[3][1] = localMatrix_TranslationAndRotation._[3][2] = 0.0f;

	localMatrix *= localMatrix_TranslationAndRotation.GetInversed();

	return Vector3(localMatrix._[0][0], localMatrix._[1][1], localMatrix._[2][2]);
}



inline Vector3 Node::GetGlobalRight()
{
	return GetGlobalRotation().GetRotatedVector(Vector3::AxisX);
}



inline Vector3 Node::GetGlobalUp()
{
	return GetGlobalRotation().GetRotatedVector(Vector3::AxisY);
}



inline Vector3 Node::GetGlobalBackward()
{
	return GetGlobalRotation().GetRotatedVector(Vector3::AxisZ);
}



inline Vector3 Node::GetGlobalForward()
{
	return -GetGlobalBackward();
}



inline Matrix Node::GetGlobalMatrix() const
{
	Matrix result = GetLocalMatrix();

	Node* tempParent = parent;
	while (tempParent != NULL)
	{
		result *= tempParent->GetLocalMatrix();
		tempParent = tempParent->parent;
	}

	return result;
}



inline Matrix Node::GetGlobalMatrix_TranslationAndRotation() const
{
	Matrix result = GetLocalMatrix_TranslationAndRotation();

	Node* tempParent = parent;
	while (tempParent != NULL)
	{
		result *= tempParent->GetLocalMatrix_TranslationAndRotation();
		tempParent = tempParent->parent;
	}

	return result;
}



inline Matrix Node::GetGlobalMatrixWithoutLocalMatrix() const
{
	Matrix result = Matrix::Identity();

	Node* tempParent = parent;
	while (tempParent != NULL)
	{
		result *= tempParent->GetLocalMatrix();
		tempParent = tempParent->parent;
	}

	return result;
}



inline void Node::ComputeGlobalBasisVectors(Vector3& right, Vector3& up, Vector3& backward) const
{
	Quaternion temp = GetGlobalRotation();

	right = temp.GetRotatedVector(Vector3::AxisX);
	up = temp.GetRotatedVector(Vector3::AxisY);
	backward = temp.GetRotatedVector(Vector3::AxisZ);
}



inline void Node::ComputeGlobalBasisVectors_Forward(Vector3& right, Vector3& up, Vector3& forward) const
{
	ComputeGlobalBasisVectors(right, up, forward);
	forward = -forward;
}



inline void Node::UpdateLocalMatrix()
{
	localMatrix = Matrix::Scale(GetLocalScale()) * GetLocalRotation().GetMatrix();
	memcpy(&localMatrix._[3][0], &GetLocalTranslation(), sizeof(Vector3));
}



#endif
