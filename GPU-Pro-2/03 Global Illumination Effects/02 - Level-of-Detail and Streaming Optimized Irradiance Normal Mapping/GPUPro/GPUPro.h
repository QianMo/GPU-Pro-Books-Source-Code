#include "ExampleApplication.h"
#include "GPUProFrameListener.h"
#include "OgreMaxScene.hpp"




using namespace OgreMax;

class GPUProApplication : public ExampleApplication
{
protected:
public:
    GPUProApplication()
    {
    }

    ~GPUProApplication() 
    {

		
    }
protected:

    GPUProFrameListener* mGPUProFrameListener;

	Ogre::RenderWindow* getRenderWindow(void) const { return mWindow; }
    

	void createFrameListener(void)
    {
		 // Create the FrameListener
        mGPUProFrameListener = new GPUProFrameListener(mWindow, mCamera, mSceneMgr);
		mFrameListener = mGPUProFrameListener; //pass down Pointer
        mRoot->addFrameListener(mFrameListener);
		mGPUProFrameListener->showDebugOverlay(true);
		mGPUProFrameListener->showHelpOverlay(true);
    }

	virtual void createCamera(void)
    {
        // Create the camera
        mCamera = mSceneMgr->createCamera("PlayerCam");
        // Position it at 500 in Z direction
        mCamera->setPosition(Vector3(0,5,50));
        // Look back along -Z
        mCamera->lookAt(Vector3(0,0,0));
        mCamera->setNearClipDistance(2);

    }
	

    void createScene(void)
    {

		//Scene load
		OgreMaxScene* Scene1 = new OgreMaxScene;
		Scene1->Load("GPUPro.scene",mWindow,0,mSceneMgr,mSceneMgr->getRootSceneNode());


		//default states=best looking states
		MaterialManager::getSingleton().setDefaultTextureFiltering(TFO_ANISOTROPIC);
		MaterialManager::getSingleton().setDefaultAnisotropy(8);

		//Material Generation 
		MaterialPtr IrrNormMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNorm");
		if (IrrNormMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNorm material not found","createScene");
        }

		MaterialPtr IrrNormNoAlbedoMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormNoAlbedo");
		if (IrrNormNoAlbedoMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormNoAlbedo material not found","createScene");
        }

		MaterialPtr IrrNormLinMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormLin");
		if (IrrNormLinMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormLin material not found","createScene");
        }

		MaterialPtr IrrNormLinNoAlbedoMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormLinNoAlbedo");
		if (IrrNormLinNoAlbedoMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormLinNoAlbedo material not found","createScene");
        }

		MaterialPtr IrrNormConstMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormConst");
		if (IrrNormConstMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormConst material not found","createScene");
        }

		MaterialPtr IrrNormConstNoAlbedoMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormConstNoAlbedo");
		if (IrrNormConstNoAlbedoMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormConstNoAlbedo material not found","createScene");
        }

		MaterialPtr IrrNormLODMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormLOD");
		if (IrrNormLODMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormLOD material not found","createScene");
        }

		MaterialPtr IrrNormLODNoAlbedoMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName("IrrNormLODNoAlbedo");
		if (IrrNormLODMaterial.isNull()) {
			 Ogre::Exception(Ogre::Exception::ERR_ITEM_NOT_FOUND,"IrrNormLODNoAlbedo material not found","createScene");
        }


		SceneManager::MovableObjectIterator EntityIterator = mSceneMgr->getMovableObjectIterator("Entity");
		Entity * currentEntity = NULL;

		while( EntityIterator.hasMoreElements() )
		{

			MaterialPtr currentMaterial;
			
			currentEntity = static_cast<Ogre::Entity *>(EntityIterator.peekNextValue());
			String currentEntityName=currentEntity->getName();

			String mayaTexture=currentEntity->getSubEntity(0)->getMaterial()->getTechnique(0)->getPass(0)->getTextureUnitState(0)->getTextureName();
			String mayaBaseName;
			String mayaExtension;
			Ogre::StringUtil::splitBaseFilename(mayaTexture,mayaBaseName,mayaExtension);
 		
		    AliasTextureNamePairList currentAlias;

			currentAlias["AlbedoMap"] = mayaTexture;
			currentAlias["NormalMap"] = mayaBaseName+"Normal."+mayaExtension;

			String currentExtension = "png";

			currentAlias["Coeff0"] = currentEntityName + "_0."+currentExtension;
			currentAlias["Coeff1"] = currentEntityName + "_1."+currentExtension;
			currentAlias["Coeff2"] = currentEntityName + "_2."+currentExtension;
			currentAlias["Coeff3"] = currentEntityName + "_3."+currentExtension;
			currentAlias["Coeff4"] = currentEntityName + "_4."+currentExtension;
			currentAlias["Coeff5"] = currentEntityName + "_5."+currentExtension;

			//full material
			currentMaterial = IrrNormMaterial->clone(currentEntityName+"IrrNormMat"); 
			currentMaterial->applyTextureAliases(currentAlias);

			//set this as the starting material
			currentEntity->setMaterial(currentMaterial);

			//noalbedo material
			currentMaterial = IrrNormNoAlbedoMaterial->clone(currentEntityName+"IrrNormNoAlbedoMat"); 
			currentMaterial->applyTextureAliases(currentAlias);

			//linear versions
			currentMaterial = IrrNormLinMaterial->clone(currentEntityName+"IrrNormLinMat"); 
			currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = IrrNormLinNoAlbedoMaterial->clone(currentEntityName+"IrrNormLinNoAlbedoMat"); 
			currentMaterial->applyTextureAliases(currentAlias);

			//lightmap versions
			currentMaterial = IrrNormConstMaterial->clone(currentEntityName+"IrrNormConstMat"); 
		    currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = IrrNormConstNoAlbedoMaterial->clone(currentEntityName+"IrrNormConstNoAlbedoMat"); 
		    currentMaterial->applyTextureAliases(currentAlias);


			//LOD material consisting of linear and lightmap
			currentMaterial = IrrNormLODMaterial->clone(currentEntityName+"IrrNormLODMat"); 
		    currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = IrrNormLODNoAlbedoMaterial->clone(currentEntityName+"IrrNormLODNoAlbedoMat"); 
		    currentMaterial->applyTextureAliases(currentAlias);
			
			EntityIterator.moveNext();
		}


		mSceneMgr->setSkyBox(true, "CampusSky", 50 );


    }
};




