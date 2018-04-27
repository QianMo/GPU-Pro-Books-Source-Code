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

#ifndef KLCONSOLE_H
#define KLCONSOLE_H

#include <map>

class klConVariable {
private:
	float fvalue;
	int ivalue;
	std::string  value;
public:
	klConVariable(const std::string &value);
	klConVariable(const char *value);

	inline int getValueInt() { return ivalue; }
	inline float getValueFloat() { return fvalue; }
	inline const char *getValue() { return value.c_str(); }
	
    void setValue(const char *value);
    void setValue(const std::string &value);

    void setValue(int newValue);
    void setValue(float newValue);
};

typedef void (* klCommandType) (const char *);

class klConsole {
private:
    typedef std::map<std::string , klConVariable *> VarMap;
	typedef std::pair<std::string , klConVariable *> VarPair;
	typedef std::map<std::string , klCommandType> ComMap;
	typedef std::pair<std::string , klCommandType> ComPair;

	VarMap variables;
	ComMap commands;
	std::string  commandBuffer;

    // Execute commands in the given stream
    void executeStream(std::istream &stream);

public:

    // Get a variable, if the variable does not exist it will automatically be created
	klConVariable *getVariable(const std::string  &name, const std::string  &value = "");

    // Register a commant with a certain name
	void registerCommand(const std::string  &name, klCommandType c);

	// Add a command to the command queue (will be executed when executeBuffer is called)
    void queueCommand(const std::string  &command);

    // Execute any queued commands
	void executeBuffer(void);

    // Exec a command right now.
	void execCommand(const std::string  &name,const std::string  &args);

    // Executes commands stored in a script file (';' separated '//' comments)
	void executeScript(const char *fileName);

    // Creates the console window and starts capturing
    // all other functionality will work but text printed before init is called
    // will be lost...
    void init(void);

    // Shows the console window
    void show(void);
};

extern klConsole console;

#endif //KLCONSOLE_H