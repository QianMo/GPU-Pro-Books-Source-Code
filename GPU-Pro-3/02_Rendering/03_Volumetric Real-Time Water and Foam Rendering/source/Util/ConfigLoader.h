#ifndef __CONGIFLOADER__H__
#define __CONGIFLOADER__H__

#include <string>

using namespace std;

#include "../Util/Singleton.h"

class ConfigLoader : public Singleton<ConfigLoader>
{
	friend class Singleton<ConfigLoader>;

public:
	ConfigLoader(void);
	~ConfigLoader(void);

	/// Loads config data from spec. file
	void LoadConfigFile(const char* filename);

	/// Get loaded screen width
	int GetScreenWidth(void) const { return screenWidth; }

	/// Get loaded screen height
	int GetScreenHeight(void) const { return screenHeight; }

	/// Get loaded bits per pixel
	int GetBitsPerPixel(void) const { return bitsPerPixel; }

	/// Get loaded refresh rate
	int GetRefreshRate(void) const { return refreshRate; }

	/// Get loaded flag for fullscreen
	int IsFullScreen(void) const { return fullScreen; }

	/// Get loaded window name (name of the game)
	const char* GetWindowName(void) const { return windowName.c_str(); }

	/// Get scene index
	unsigned int GetSceneIndex(void) const { return sceneIndex; }
	
private:
	/// Screen width
	int screenWidth;

	/// Screen height
	int screenHeight;

	/// Bits per pixel
	int bitsPerPixel;

	/// Refresh rate
	int refreshRate;

	/// Fullscreen flag
	int fullScreen;

	/// Window name
	string windowName;

	/// Start scene index
	unsigned int sceneIndex;
};
#endif