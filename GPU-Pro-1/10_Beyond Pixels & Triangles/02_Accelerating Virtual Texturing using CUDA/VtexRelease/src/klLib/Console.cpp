#include "shared.h"

#include "Console.h"
#include "FileSystem.h"
#include "ConsoleWindow.h"

//------------ klConVariable ------------------

klConVariable::klConVariable(const char *value) {
	setValue(value);
}

klConVariable::klConVariable(const std::string &value) {
	setValue(value);
}

void klConVariable::setValue(const std::string &value) {
 	this->value = value;
    ivalue = atoi(this->value.c_str());
    fvalue = (float)atof(this->value.c_str());
}
void klConVariable::setValue(const char *value) {
	this->value = value;
	ivalue = atoi(value);
	fvalue = (float)atof(value);
}

void klConVariable::setValue(int newValue) {
    ivalue = newValue;
    fvalue = (float)newValue;
    char buff[256];
    sprintf(buff,"%i",newValue);
    this->value = buff;
}

void klConVariable::setValue(float newValue) {
    ivalue = (int)newValue;
    fvalue = newValue;
    char buff[256];
    sprintf(buff,"%f",newValue);
    this->value = buff;
}

//------------ klConsole ------------------

klConVariable *klConsole::getVariable(const std::string  &name, const std::string  &value) {
	klConVariable *v = variables[name];
	if (!v) {
		v = new klConVariable(value);
		variables[name] = v;
	}
	return v;
}

void klConsole::registerCommand(const std::string  &name, klCommandType c) {
	if (!commands[name]) {
		commands[name] = c;
	}
}

void klConsole::queueCommand(const std::string  &command) {
	commandBuffer.append(command+";");
}

void klConsole::execCommand(const std::string  &name,const std::string  &args) {

	//Check for system commands
	if (name == "toggle") {
		klConVariable *v = variables[args];
		if (!v) {
			klLog(("Variable not found: "+name).c_str());
			return;
		}

		if (v->getValueInt())
			execCommand(args, std::string ("0"));
		else
			execCommand(args, std::string ("1"));
		return;
	}

	//Check for user commands
	klCommandType c = commands[name];
	if (c) {
		(*c)(args.c_str());
		return;
	}

	//Check for user variables
	klConVariable *v = getVariable(name,"");//variables[name];
	if (!v) {
		klLog(("Command not found: "+name).c_str());
		return;
	}

    // If no new value just echo
	if (args == "")
		klLog((name + " = " + v->getValue()).c_str());
	else
		v->setValue(args);
}

void klConsole::executeBuffer(void) {
    std::istringstream ss(commandBuffer);
	executeStream(ss);
}

void klConsole::executeScript(const char *fileName) {
    std::istream *is = fileSystem.openFile(fileName);
    executeStream(*is);
    delete is;
}

void klConsole::executeStream(std::istream &stream) {
    char token[256];
    klStringStream str(stream);
	
    while ( str.getToken(token) ) {
        std::string  command = token;
        std::string  arguments;

        while ( str.getToken(token) ) {
            if ( token[0] == ';' ) {
                break;
            } else {
                arguments += " ";
                arguments += token;
            }
        }

        execCommand(command,arguments);
    }
}

class klConsoleWindowListener : public ConsoleWindow::Listener {
    virtual void command(const char *text) {
        std::string str(text);
        size_t space = str.find_first_of(' ');
        if ( space == std::string::npos ) {
            console.execCommand(str,"");
        } else {
            std::string name = str.substr(0,space);
            std::string args = str.substr(space+1,str.size()-space-1);
            console.execCommand(name.c_str(),args.c_str());
        }
    }    
};

static klConsoleWindowListener klConListener;

void klConsole::init(void) {
    ConsoleWindow::Create(&klConListener);
}

void klConsole::show(void) {
    ConsoleWindow::Show();   
}

klConsole console;