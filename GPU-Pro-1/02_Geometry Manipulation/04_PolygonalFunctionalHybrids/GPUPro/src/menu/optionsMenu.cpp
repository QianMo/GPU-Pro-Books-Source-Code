
#if defined(__APPLE__) || defined(MACOSX)
   #include <GLUT/glut.h>
#else
   #include <GL/glut.h>
#endif


#include "optionsMenu.h"

MENU& getMenu()
{
	static MENU menu;

	return menu;
}

MENU::MENU() : lastGeneratedId(ITEM_CUSTOM_FIRST), itemCustomLast(-1)
{

}

// very basic handling of the menu choices
bool MENU::menuModifyValues(int menuItemCurrent, int menuItemBegin, int menuItemEnd, MENU::CHOICE_MODE choiceMode)
{
   if (menuItemCurrent > menuItemEnd || menuItemBegin < menuItemBegin) {
      return false;
   }

   if (choiceMode == MULTIPLE_CHOICES_OFF) {

      // let only one option be enabled
      for (int i = menuItemBegin; i <= menuItemEnd; i++) {			
			getMenu().setValue((MENU::ITEMS)i, false);
      }

      getMenu().setValue((MENU::ITEMS)menuItemCurrent, true);

   } else {
      
      getMenu().toggleValue((MENU::ITEMS)menuItemCurrent);

   }

   return true;
}

// generate unique ids for menu items
int MENU::getCustomId(REQUEST_TYPE type)
{
   return type == REQUEST_NEW ? lastGeneratedId++ : lastGeneratedId ;
}


// return in index of the newly created submenu having a set of custom items
int MENU::addCustomItems(const LIST_DESC& customItems, MENU::GROUP groupIndex)
{
   int menuId = glutCreateMenu(customHandler);
   
   ITEMS_RANGE &range = itemRanges[groupIndex];

   range.minId = getCustomId(REQUEST_CURRENT);

   for (int i = 0; i < customItems.itemsSize; i++ ) {
      glutAddMenuEntry(customItems.itemNames[i], getCustomId());
   }

   range.maxId = getCustomId(REQUEST_CURRENT);

   return menuId;
}

void MENU::init(LIST_DESC scenes, LIST_DESC textures, LIST_DESC cubemaps, LIST_DESC resolutions)
{
   // some default values:

   for (int i = ITEM_MAIN_BEGIN; i < ITEM_MAIN_END; i++) {
      values[i] = false;
   }
   
   values[ITEM_TOGGLE_ANIMATION]   =  true;
   values[ITEM_TOGGLE_MESH]   =  true;
   values[ITEM_TOGGLE_FREP]   =  true;
   //values[ITEM_TOGGLE_SKELETON]    =  false;
   //values[ITEM_DIFFUSE_SHADING]    =  false;
   //values[ITEM_CUBEMAP_SHADING]    =  false;
   values[ITEM_PROCEDURAL_SHADING] =  true;

   getItem(GROUP_RESOLUTION).current = 1;

   // using basic glut UI here for simplicity

   int visibilitySubmenu = glutCreateMenu( menuHandler<ITEM_TOGGLE_MESH, ITEM_TOGGLE_SKELETON, MULTIPLE_CHOICES_ON> );
   glutAddMenuEntry("Mesh", ITEM_TOGGLE_MESH);
   glutAddMenuEntry("FRep", ITEM_TOGGLE_FREP);	 
   glutAddMenuEntry("Skeleton", ITEM_TOGGLE_SKELETON);
   
   int shadingSubmenu = glutCreateMenu( menuHandler<ITEM_CUBEMAP_SHADING, ITEM_TRIPLANAR_SHADING, MULTIPLE_CHOICES_OFF> );

   glutAddMenuEntry("Cubemap", ITEM_CUBEMAP_SHADING);
   glutAddMenuEntry("Procedural", ITEM_PROCEDURAL_SHADING);
   glutAddMenuEntry("Triplanar texturing", ITEM_TRIPLANAR_SHADING);

   int sceneSubmenu     =  addCustomItems(scenes, GROUP_SCENE);
   int textureSubmenu   =  addCustomItems(textures, GROUP_TEXTURE);
   int cubemapSubmenu   =  addCustomItems(cubemaps, GROUP_CUBEMAP);
   int resolutionSubmenu=  addCustomItems(resolutions, GROUP_RESOLUTION);

   glutCreateMenu( menuHandler<ITEM_TOGGLE_ANIMATION, ITEM_TOGGLE_WIREFRAME, MULTIPLE_CHOICES_ON> );

   glutAddMenuEntry("Toggle animation", ITEM_TOGGLE_ANIMATION);
   glutAddMenuEntry("Toggle wireframe", ITEM_TOGGLE_WIREFRAME);

   glutAddSubMenu("Scenes", sceneSubmenu);
   glutAddSubMenu("Resolution", resolutionSubmenu);
   glutAddSubMenu("Show/Hide", visibilitySubmenu);
   glutAddSubMenu("Shading", shadingSubmenu);
   glutAddSubMenu("Texture", textureSubmenu);
   glutAddSubMenu("Cubemap", cubemapSubmenu);
   

   glutAddMenuEntry("Quit (esc)", '\033');
   glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void MENU::customHandler(int item)
{
	if (item < MENU::ITEM_CUSTOM_FIRST) {
      return;
   }

	for (int i = 0; i < GROUP_LAST; i++ ) {
      getMenu().getItem((MENU::GROUP)i).setIfInRange(item);
   }

}
