#include <node.hpp>



Node::Node()
{
	name = "Unknown";

	localTranslation = Vector3(0.0f);
	localRotation = Quaternion::Identity();
	localScale = Vector3(1.0f);

	UpdateLocalMatrix();
}



void Node::SetParent(Node* newParent)
{
	if (parent == newParent)
	{
		return;
	}
	else if (parent != NULL)
	{
		for (uint i = 0; i < parent->children.size(); i++)
		{
			if (parent->children[i] == this)
			{
				parent->children.erase(parent->children.begin() + i);
				break;
			}
		}
	}

	if (newParent)
		newParent->children.push_back(this);

	// determine local transformation with respect to the new parent (so global transformation in most cases will remain the same after the parent has been changed)
	{
		Matrix localMatrix = GetLocalMatrix();
		Matrix localMatrix_translationAndRotation = GetLocalMatrix_TranslationAndRotation();

		Node* tempParent;

		// up from the current parent
		tempParent = parent;
		while (tempParent != NULL)
		{
			localMatrix *= tempParent->GetLocalMatrix();
			localMatrix_translationAndRotation *= tempParent->GetLocalMatrix_TranslationAndRotation();

			tempParent = tempParent->parent;
		}

		// down to the new parent composition
		tempParent = newParent;
		Matrix tempMatrix = Matrix::Identity();
		Matrix tempMatrix_translationAndRotation = Matrix::Identity();
		while (tempParent != NULL)
		{
			tempMatrix *= tempParent->GetLocalMatrix();
			tempMatrix_translationAndRotation *= tempParent->GetLocalMatrix_TranslationAndRotation();

			tempParent = tempParent->parent;
		}
		localMatrix *= tempMatrix.GetInversed();
		localMatrix_translationAndRotation *= tempMatrix_translationAndRotation.GetInversed();

		//

		this->localTranslation = Vector3(localMatrix._[3][0], localMatrix._[3][1], localMatrix._[3][2]);

		// remove translation parts
		localMatrix._[3][0] = localMatrix._[3][1] = localMatrix._[3][2] = 0.0f;
		localMatrix_translationAndRotation._[3][0] = localMatrix_translationAndRotation._[3][1] = localMatrix_translationAndRotation._[3][2] = 0.0f;

		// leave scale and skew only
		localMatrix *= localMatrix_translationAndRotation.GetInversed();

		this->localRotation = localMatrix_translationAndRotation.GetQuaternion();
		this->localScale = Vector3(localMatrix._[0][0], localMatrix._[1][1], localMatrix._[2][2]);

		UpdateLocalMatrix();
	}

	//

	parent = newParent;
}
