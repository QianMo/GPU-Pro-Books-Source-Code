#ifndef _OPTIONS_MENU_H_
#define _OPTIONS_MENU_H_

#include <string>

#include "../defines.h"

// structure used to work with very basic GLUT-based menu (creation, handling, params etc)
class MENU {
public:

   static const int ITEMS_CHOICE_MAX = 64;

   struct ITEMS_RANGE {

      int   getSize  () const { return maxId - minId; }
      bool  isInRange( int index ) const { return index >= minId && index < maxId; }
      int   getRelative ( int index ) const { return index - minId; }
      bool  setIfInRange (int index) 
      {
         previous = current;

         if (isInRange(index)) {             
            current = getRelative(index); 
            return true; 
         } 

         return false;
      }

      // = true, if the value was recently changed 
      bool isChanged() const { return previous != current;}
      // to reset once the changed state has been retrieved
      void resetChanged() { previous = current; }

      //int   getValue () const { return current; }

      operator int () const { return current; }

      int minId, maxId, current, previous;
   };
   
   // glut menu options
   enum CHOICE_MODE {
      MULTIPLE_CHOICES_OFF = 0,
      MULTIPLE_CHOICES_ON
   };
   
   enum ITEMS {
      
      ITEM_MAIN_BEGIN = 127,

         ITEM_TOGGLE_ANIMATION,
         ITEM_TOGGLE_WIREFRAME,

         ITEM_TOGGLE_MESH,
         ITEM_TOGGLE_FREP,
         ITEM_TOGGLE_SKELETON,

         ITEM_DIFFUSE_SHADING,
         ITEM_CUBEMAP_SHADING,
         ITEM_PROCEDURAL_SHADING,
         ITEM_TRIPLANAR_SHADING,  

         ITEM_QUIT,

      ITEM_MAIN_END,

      // all custom items will be added after this item 
      ITEM_CUSTOM_FIRST,   
   };

  enum GROUP {
      GROUP_TEXTURE = 0,
      GROUP_CUBEMAP,
      GROUP_SCENE,
      GROUP_RESOLUTION,

		GROUP_LAST
   };

   enum REQUEST_TYPE {
      REQUEST_NEW,
      REQUEST_CURRENT
   };

	MENU();

	// build menus provided some external description
   void init(LIST_DESC scenes, LIST_DESC textures, LIST_DESC cubemaps, LIST_DESC resolutions);

   // generate unique ids for menu items
   int getCustomId(REQUEST_TYPE type = REQUEST_NEW);

	bool getValue		(ITEMS itemIndex) const { return values[itemIndex]; }
	bool setValue		(ITEMS itemIndex, bool value)	{ return values[itemIndex] = value; }
	bool toggleValue	(ITEMS itemIndex)	{ return values[itemIndex] = !values[itemIndex]; }

	ITEMS_RANGE& getItem (GROUP group) { return itemRanges[group]; }

private:

   // return in index of the newly created submenu having a set of custom items
   int   addCustomItems(const LIST_DESC& customItems, GROUP groupIndex);

	// these methods are static as they are used from glut :
   static void customHandler(int item);

	static bool menuModifyValues(int menuItemCurrent, int menuItemBegin, int menuItemEnd, MENU::CHOICE_MODE choiceMode);

	// this template function is used to generate custom functions handling changes in the menu
	template <int ITEM_BEGIN, int ITEM_END, CHOICE_MODE MULTIPLE_CHOICES_MODE>
	static void menuHandler(int menuItem)
	{
		menuModifyValues(menuItem, ITEM_BEGIN, ITEM_END, MULTIPLE_CHOICES_MODE);
	}

	bool values[ITEM_MAIN_END];

   // some custom menu items
   ITEMS_RANGE itemRanges[GROUP_LAST];

	int	itemCustomLast;
	int	lastGeneratedId;
};

MENU& getMenu();

#endif // _OPTIONS_MENU_H_
