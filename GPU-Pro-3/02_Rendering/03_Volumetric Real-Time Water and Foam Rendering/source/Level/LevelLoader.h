#ifndef __LEVELLOADER__H__
#define __LEVELLOADER__H__

#include <string>
#include <map>
#include <list>
using namespace std;

#include "../Util/Singleton.h"
#include "../Util/Vector3.h"
#include "../XMLParser/XmlNotify.h"
#include "../Physic/Physic.h"

class RenderNode;
class Level;

class LevelLoader : public Singleton<LevelLoader>, XmlNotify
{
	friend class Singleton<LevelLoader>;

private:
	// structure for hold loaded data
	struct gameElement 
	{
		string type;
		Vector3 position;
		Vector3 rotation;
		string meshFileName;
		std::vector<string> collFileNames;
		Vector3 measures;
		float density;
		float radius;
		int precision;
		Vector3 uTex;
		Vector3 vTex;
		int	material;
	};

public:
	LevelLoader(void);
	~LevelLoader(void);

	// loads the level with spec. filename
	void LoadLevel(const char* fileName, bool createJoint);

	// callback functions for xml parser
	void foundNode		(string& name, string& attributes);
	void foundElement	(string& name, string& value, string& attributes);
	void startElement	(string& name, string& value, string& attributes);
	void endElement		(string& name, string& value, string& attributes);

	const Vector3& GetCameraPosition(void) const { return cameraPosition; }
	const Vector3& GetCameraDirection(void) const { return cameraDirection; }

private:
	// creates the level tree that is given to the render manager
	void createLevelTree(list<gameElement*>& geList, list<Physic::physicData*>& childs);

	void destroyGameElements(list<gameElement*> gE);

	void destroyPhysicsData(list<Physic::physicData*> pD);

	// possible node types in a level xml
	enum NODETYPES
	{
		NODETYPE_NONE = 0,
		NODETYPE_LEVEL,
		NODETYPE_GAMEELEMENT,
		NODETYPE_ELEMENTPOSITION,
		NODETYPE_ELEMENTROTATION,
		NODETYPE_MESHFILENAME,
		NODETYPE_COLLFILENAME,
		NODETYPE_DENSITY,
		NODETYPE_RADIUS,
		NODETYPE_PRECISION,
		NODETYPE_ELEMENTMEASURES,
		NODETYPE_ELEMENTUTEX,
		NODETYPE_ELEMENTVTEX,
		NODETYPE_MATERIAL,
		NODETYPE_TEXTUREFILENAMES
	};

	// skybox textures
	string skyBoxTextureFileNames[6];

	// current data during xml parsing
	NODETYPES currentNode;
	gameElement* currentGameElement;

	// loaded data
	list<gameElement*> gameElements;
	list<gameElement*>::iterator iter;

	// initial camera position and direction
	Vector3 cameraPosition;
	Vector3 cameraDirection;
};

#endif