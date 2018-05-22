
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "Platform.h"
#include "Util/Array.h"

struct Entry;

class Config {
public:
	Config();
	~Config();

	bool init(const char *subPath = NULL);
	bool flush();

	bool getBoolDef(const char *name, const bool def) const;
	int getIntegerDef(const char *name, const int def) const;
	float getFloatDef(const char *name, const float def) const;
	bool getInteger(const char *name, int &dest) const;
	void setBool(const char *name, const bool val);
	void setInteger(const char *name, const int val);
	void setFloat(const char *name, const float val);
private:

	Array <Entry> entries;
	char *path;
};


#endif // _CONFIG_H_
