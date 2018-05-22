
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

#include "Config.h"

#ifndef _WIN32
#  include "Util/Tokenizer.h"
#  include <sys/stat.h>
#endif

#include <stdio.h>

struct Entry {
	Entry(const char *eName, const int val, const bool dirt){
		name = new char[strlen(eName) + 1];
		strcpy(name, eName);
		value = val;
		dirty = dirt;
	}
	Entry(const char *eName, const float val, const bool dirt){
		name = new char[strlen(eName) + 1];
		strcpy(name, eName);
		floatValue = val;
		dirty = dirt;
	}

	char *name;
	union {
		int value;
		float floatValue;
	};
	bool dirty;
};

Config::Config(){
	path = NULL;
}

Config::~Config(){
	for (unsigned int i = 0; i < entries.getCount(); i++){
		delete entries[i].name;
	}
	delete path;
}

bool Config::init(const char *subPath){
	size_t len = subPath? strlen(subPath) : 0;

#ifdef _WIN32
	path = new char[len + 16];
	strcpy(path, "SOFTWARE\\Humus");
	if (len){
		strcat(path + 14, "\\%s");
	}

	HKEY hkey;
	if (RegOpenKeyEx(HKEY_CURRENT_USER, path, 0, KEY_READ, &hkey) == ERROR_SUCCESS){
		char name[256];
		unsigned long type, nameSize = sizeof(name);
		unsigned long value, valueSize;

		unsigned int i = 0;
		while (RegEnumValue(hkey, i, name, &nameSize, NULL, &type, NULL, NULL) != ERROR_NO_MORE_ITEMS){
			if (type == REG_DWORD){
				valueSize = sizeof(value);
				if (RegQueryValueEx(hkey, name, NULL, NULL, (LPBYTE) &value, &valueSize) == ERROR_SUCCESS){
					entries.add(Entry(name, (int) value, false));
				}
			}
			nameSize = sizeof(name);
			i++;
		}
		RegCloseKey(hkey);
		return true;
	}

#else

	char *home = getenv("HOME");

	path = new char[len + strlen(home) + 23];
	int pos = sprintf(path, "%s/.humus", home);
	mkdir(path, 0777);
	if (len){
		pos += sprintf(path + pos, "/%s", subPath);
		mkdir(path, 0777);
	}
	strcpy(path + pos, "/settings.conf");

	Tokenizer tok(2);
	if (tok.setFile(path)){
		char *name, *value;
		while ((name = tok.next()) != NULL){
			if ((value = tok.nextAfterToken("=")) != NULL){
				if ((name[0] >= 'A' && name[0] <= 'Z') || (name[0] >= 'a' && name[0] <= 'z')){
					entries.add(Entry(name, atoi(value), false));
				}
				tok.goToNext();
			}
		}

		return true;
	}
	
#endif

	return false;
}

bool Config::flush(){
	bool created = false;

#ifdef _WIN32

	HKEY hkey;

	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (entries[i].dirty){
			if (!created){
				if (RegCreateKeyEx(HKEY_CURRENT_USER, path, NULL, "REG_SZ", REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hkey, NULL) != ERROR_SUCCESS) return false;
				created = true;
			}
			RegSetValueEx(hkey, entries[i].name, 0, REG_DWORD, (unsigned char *) &entries[i].value, sizeof(int));
		}
	}
	if (created) RegCloseKey(hkey);

#else

	FILE *file = NULL;
	unsigned int i = 0;
	while (i < entries.getCount()){
		if (created){
			fprintf(file, "%s = %d;\n", entries[i].name, entries[i].value);
		} else {
			if (entries[i].dirty){
				file = fopen(path, "wb");
				created = true;
				i = 0;
				continue;
			}
		}
		i++;
	}
	if (created) fclose(file);

#endif
	return true;
}

bool Config::getBoolDef(const char *name, const bool def) const {
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			return (entries[i].value != 0);
		}
	}
	return def;
}

int Config::getIntegerDef(const char *name, const int def) const {
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			return entries[i].value;
		}
	}
	return def;
}

float Config::getFloatDef(const char *name, const float def) const {
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			return entries[i].floatValue;
		}
	}
	return def;
}

bool Config::getInteger(const char *name, int &dest) const {
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			dest = entries[i].value;
			return true;
		}
	}
	return false;
}

void Config::setBool(const char *name, bool val){
	setInteger(name, val);
}

void Config::setInteger(const char *name, int val){
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			if (entries[i].value != val){
				entries[i].value = val;
				entries[i].dirty = true;
			}
			return;
		}
	}
	entries.add(Entry(name, val, true));
}

void Config::setFloat(const char *name, const float val){
	for (unsigned int i = 0; i < entries.getCount(); i++){
		if (stricmp(entries[i].name, name) == 0){
			if (entries[i].floatValue != val){
				entries[i].floatValue = val;
				entries[i].dirty = true;
			}
			return;
		}
	}
	entries.add(Entry(name, val, true));
}
