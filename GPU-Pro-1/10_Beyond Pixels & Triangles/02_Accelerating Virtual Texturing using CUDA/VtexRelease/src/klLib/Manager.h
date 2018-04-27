/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#ifndef KLMANAGER_H
#define KLMANAGER_H

#include <map>

template <class Type> class klManager {
protected:
    std::map<std::string, Type *>objects;

	/**
	Clients should override this
	*/
	virtual Type *getInstance(const char *name) = 0;

public:
	klManager(void) {
        objects = std::map<std::string, Type *>();
		objects.clear();
	}

	Type *getForName(const char *name) {
		Type *t = objects[name];
		if (!t ) {
			t = getInstance(name);
			objects[name] = t;
		}
		return t;
	}

	void freeAll(void) {
        for (std::map<std::string, Type *>::iterator i = objects.begin(); i != objects.end(); i++) {
			delete i->second;
			i->second = NULL;
		}
		objects.clear();
	}

    void reload(void) {
        for (std::map<std::string, Type *>::iterator i = objects.begin(); i != objects.end(); i++) {
            i->second->reload(i->first.c_str());
		}
    }

	virtual ~klManager(void) {
		freeAll();
	}
};

#endif //KLMANAGER_H