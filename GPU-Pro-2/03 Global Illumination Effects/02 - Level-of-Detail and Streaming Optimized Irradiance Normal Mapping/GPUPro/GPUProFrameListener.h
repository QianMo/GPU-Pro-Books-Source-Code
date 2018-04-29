#include "HDRListener.h"

class GPUProFrameListener : public ExampleFrameListener
{
public:
    GPUProFrameListener(RenderWindow* win, Camera* cam, SceneManager *sceneMgr)
      : ExampleFrameListener(win, cam, false, false),
		mSceneMgr(sceneMgr),
		mHelpOn(true),
		mHelpOverlay(0),
		mAlbedoOn(true),
		mHDROn(false),
		mBasisType(QUADRATIC),
		mTextureType(PNG),
		hdrListener(0)
    {

		mHelpOverlay = OverlayManager::getSingleton().getByName("HelpOverlay");

		registerCompositors();

    }
		
	virtual ~GPUProFrameListener()
	{

		delete hdrListener;

		//Remove ourself as a Window listener
		WindowEventUtilities::removeWindowEventListener(mWindow, this);
		windowClosed(mWindow);
	}

	enum BasisType
	{
		CONSTANT, 
		LINEAR, 
		QUADRATIC,
		LOD
	};
	enum TextureType
	{
		PNG, 
		DXT, 
		EXR
	};




	void showHelpOverlay(bool show)
	{
		if (mHelpOverlay)
		{
			if (show)
				mHelpOverlay->show();
			else
				mHelpOverlay->hide();
		}
	}

    // Overriding the default processUnbufferedKeyInput so the key updates we define
    // later on work as intended.
    bool processUnbufferedKeyInput(const FrameEvent& evt)
	{

		if(mKeyboard->isKeyDown(OIS::KC_A))
			mTranslateVector.x = -mMoveScale;	// Move camera left

		if(mKeyboard->isKeyDown(OIS::KC_D))
			mTranslateVector.x = mMoveScale;	// Move camera RIGHT

		if(mKeyboard->isKeyDown(OIS::KC_UP) || mKeyboard->isKeyDown(OIS::KC_W) )
			mTranslateVector.z = -mMoveScale;	// Move camera forward

		if(mKeyboard->isKeyDown(OIS::KC_DOWN) || mKeyboard->isKeyDown(OIS::KC_S) )
			mTranslateVector.z = mMoveScale;	// Move camera backward

		if(mKeyboard->isKeyDown(OIS::KC_PGUP))
			mTranslateVector.y = mMoveScale;	// Move camera up

		if(mKeyboard->isKeyDown(OIS::KC_PGDOWN))
			mTranslateVector.y = -mMoveScale;	// Move camera down

		if(mKeyboard->isKeyDown(OIS::KC_RIGHT))
			mCamera->yaw(-mRotScale);

		if(mKeyboard->isKeyDown(OIS::KC_LEFT))
			mCamera->yaw(mRotScale);

		if( mKeyboard->isKeyDown(OIS::KC_ESCAPE) || mKeyboard->isKeyDown(OIS::KC_Q) )
			return false;

       	if( mKeyboard->isKeyDown(OIS::KC_F2) && mTimeUntilNextToggle <= 0 )
		{
			mStatsOn = !mStatsOn;
			showDebugOverlay(mStatsOn);
			mTimeUntilNextToggle = 0.5;
		}

		if( mKeyboard->isKeyDown(OIS::KC_F1) && mTimeUntilNextToggle <= 0 )
		{
			mHelpOn = !mHelpOn;
			showHelpOverlay(mHelpOn);
			mTimeUntilNextToggle = 0.5;
		}

		if( mKeyboard->isKeyDown(OIS::KC_T) && mTimeUntilNextToggle <= 0 )
		{
			switch(mFiltering)
			{
			case TFO_BILINEAR:
				mFiltering = TFO_TRILINEAR;
				mAniso = 1;
				break;
			case TFO_TRILINEAR:
				mFiltering = TFO_ANISOTROPIC;
				mAniso = 8;
				break;
			case TFO_ANISOTROPIC:
				mFiltering = TFO_BILINEAR;
				mAniso = 1;
				break;
			default: break;
			}
			MaterialManager::getSingleton().setDefaultTextureFiltering(mFiltering);
			MaterialManager::getSingleton().setDefaultAnisotropy(mAniso);

			showDebugOverlay(mStatsOn);
			mTimeUntilNextToggle = 0.5;
		}

		if(mKeyboard->isKeyDown(OIS::KC_SYSRQ) && mTimeUntilNextToggle <= 0)
		{
			std::ostringstream ss;
			ss << "screenshot_" << ++mNumScreenShots << ".png";
			mWindow->writeContentsToFile(ss.str());
			mTimeUntilNextToggle = 0.5;
			mDebugText = "Saved: " + ss.str();
		}

		if(mKeyboard->isKeyDown(OIS::KC_R) && mTimeUntilNextToggle <=0)
		{
			mSceneDetailIndex = (mSceneDetailIndex+1)%3 ;
			switch(mSceneDetailIndex) {
				case 0 : mCamera->setPolygonMode(PM_SOLID); break;
				case 1 : mCamera->setPolygonMode(PM_WIREFRAME); break;
				case 2 : mCamera->setPolygonMode(PM_POINTS); break;
			}
			mTimeUntilNextToggle = 0.5;
		}

		static bool displayCameraDetails = false;
		if(mKeyboard->isKeyDown(OIS::KC_P) && mTimeUntilNextToggle <= 0)
		{
			displayCameraDetails = !displayCameraDetails;
			mTimeUntilNextToggle = 0.5;
			if (!displayCameraDetails)
				mDebugText = "";
		}
		// Print camera details
		if(displayCameraDetails)
			mDebugText = "P: " + StringConverter::toString(mCamera->getDerivedPosition()) +
						 " " + "O: " + StringConverter::toString(mCamera->getDerivedOrientation());

		if( mKeyboard->isKeyDown(OIS::KC_TAB) && mTimeUntilNextToggle <= 0 )
		{
			mAlbedoOn = !mAlbedoOn;

			changeMaterial(mBasisType,mAlbedoOn);

			mTimeUntilNextToggle = 0.5;
		}
		
		
		if( mKeyboard->isKeyDown(OIS::KC_H) && mTimeUntilNextToggle <= 0 )
		{
			mHDROn = !mHDROn;
		
			Ogre::Viewport *vp = mWindow->getViewport(0);
			Ogre::CompositorManager::getSingleton().setCompositorEnabled(vp, "HDR", mHDROn);

			mTimeUntilNextToggle = 0.5;
		}


		if( mKeyboard->isKeyDown(OIS::KC_1) && mTimeUntilNextToggle <= 0 )
		{
			if(mBasisType != QUADRATIC)
			{
				mBasisType = QUADRATIC;
				changeMaterial(mBasisType,mAlbedoOn);
			}

		}
		if( mKeyboard->isKeyDown(OIS::KC_2) && mTimeUntilNextToggle <= 0 )
		{
			if(mBasisType!= LINEAR)
			{
				mBasisType = LINEAR;
				changeMaterial(mBasisType,mAlbedoOn);
			}

		}
		if( mKeyboard->isKeyDown(OIS::KC_3) && mTimeUntilNextToggle <= 0 )
		{
			if(mBasisType != CONSTANT)
			{
				mBasisType = CONSTANT;
				changeMaterial(mBasisType,mAlbedoOn);
			}

		}
		if( mKeyboard->isKeyDown(OIS::KC_4) && mTimeUntilNextToggle <= 0 )
		{
			if(mBasisType != LOD)
			{
				mBasisType = LOD;
				changeMaterial(mBasisType,mAlbedoOn);
			}

		}
		if( mKeyboard->isKeyDown(OIS::KC_U) && mTimeUntilNextToggle <= 0 )
		{
			
			switch(mTextureType)
			{
			case PNG:
				mTextureType = DXT;
				break;
			case DXT:
				mTextureType = EXR;
				break;
			case EXR:
				mTextureType = PNG;
			default: break;
			}
			
			setIrrMaps(mTextureType);


			mTimeUntilNextToggle = 0.5;
		
		}
		// Return true to continue rendering
		return true;
	}

    // Overriding the default processUnbufferedMouseInput so the Mouse updates we define
    // later on work as intended. 
    bool processUnbufferedMouseInput(const FrameEvent& evt)
    {
		 return ExampleFrameListener::processUnbufferedMouseInput(evt);   
    }

    bool frameStarted(const FrameEvent &evt)
    {
        return ExampleFrameListener::frameStarted(evt);
    }

	bool setIrrMaps(const TextureType tp)
	{
		
		SceneManager::MovableObjectIterator EntityIterator = mSceneMgr->getMovableObjectIterator("Entity");
		Entity * currentEntity = NULL;

		while( EntityIterator.hasMoreElements() )
		{

			MaterialPtr currentMaterial;
			
			currentEntity = static_cast<Ogre::Entity *>(EntityIterator.peekNextValue());
			String currentEntityName=currentEntity->getName();
	
			String currentExtension;
			if(tp == PNG) currentExtension = "png";
			else if (tp == DXT) currentExtension = "DDS";
			else if (tp == EXR) currentExtension = "exr";

			AliasTextureNamePairList currentAlias;

			currentAlias["Coeff0"] = currentEntityName + "_0."+currentExtension;
			currentAlias["Coeff1"] = currentEntityName + "_1."+currentExtension;
			currentAlias["Coeff2"] = currentEntityName + "_2."+currentExtension;
			currentAlias["Coeff3"] = currentEntityName + "_3."+currentExtension;
			currentAlias["Coeff4"] = currentEntityName + "_4."+currentExtension;
			currentAlias["Coeff5"] = currentEntityName + "_5."+currentExtension;

			//full material
			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormMat");
			currentMaterial->applyTextureAliases(currentAlias);

			//noalbedo material
			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormNoAlbedoMat");
			currentMaterial->applyTextureAliases(currentAlias);

			//linear versions
			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormLinMat");
			currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormLinNoAlbedoMat");
			currentMaterial->applyTextureAliases(currentAlias);

			//lightmap versions
			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormConstMat");
		    currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormConstNoAlbedoMat");
		    currentMaterial->applyTextureAliases(currentAlias);

			//LOD material consisting of linear and lightmap
			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormLODMat");
		    currentMaterial->applyTextureAliases(currentAlias);

			currentMaterial = MaterialManager::getSingleton().getByName(currentEntityName+"IrrNormLODNoAlbedoMat");
		    currentMaterial->applyTextureAliases(currentAlias);
			
			EntityIterator.moveNext();
		}




		return true;
	}


protected:

	SceneManager* mSceneMgr;

	Overlay* mHelpOverlay;

	HDRListener *hdrListener;
	
	bool mHelpOn;
	bool mAlbedoOn;
	TextureType mTextureType;
	bool mHDROn;


	BasisType mBasisType;


	bool changeMaterial(const BasisType rend,const bool albedo)
	{

		SceneManager::MovableObjectIterator EntityIterator = mSceneMgr->getMovableObjectIterator("Entity");
		Entity * currentEntity = NULL;

		while( EntityIterator.hasMoreElements() )
		{
			
			currentEntity = static_cast<Ogre::Entity *>(EntityIterator.peekNextValue());
			String currentEntityName=currentEntity->getName();

			String materialName = currentEntityName;

			if(albedo)
			{
				if(rend == CONSTANT)
				{	
					materialName += "IrrNormConstMat";
				}
				else if(rend == LINEAR)
				{	
					materialName += "IrrNormLinMat";
				}
				else if(rend == QUADRATIC)
				{	
					materialName += "IrrNormMat";
				}
				else if(rend == LOD)
				{	
					materialName += "IrrNormLODMat";
				}
			}
			else
			{
				if(rend == CONSTANT)
				{	
					materialName += "IrrNormConstNoAlbedoMat";
				}
				else if(rend == LINEAR)
				{	
					materialName += "IrrNormLinNoAlbedoMat";
				}
				else if(rend == QUADRATIC)
				{	
					materialName += "IrrNormNoAlbedoMat";
				}
				else if(rend == LOD)
				{	
					materialName += "IrrNormLODNoAlbedoMat";
				}

			}

			MaterialPtr toAssignMaterial = Ogre::MaterialManager::getSingletonPtr()->getByName(materialName);

			currentEntity->setMaterial(toAssignMaterial);
		

			EntityIterator.moveNext();
		}

	return true;

	}

	void registerCompositors(void)
	{

		hdrListener = new HDRListener();

		Ogre::Viewport *vp = mWindow->getViewport(0);

		Ogre::CompositorInstance *compInstance = Ogre::CompositorManager::getSingleton().addCompositor(vp, "HDR", -1);
		Ogre::CompositorManager::getSingleton().setCompositorEnabled(vp, "HDR", false);

		compInstance->addListener(hdrListener);
		hdrListener->notifyViewportSize(vp->getActualWidth(), vp->getActualHeight());
		hdrListener->notifyCompositor(compInstance);


	}

};
