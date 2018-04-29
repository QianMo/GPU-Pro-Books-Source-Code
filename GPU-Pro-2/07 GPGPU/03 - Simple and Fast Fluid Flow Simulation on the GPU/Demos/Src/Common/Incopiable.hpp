#ifndef INCOPIABLE_H
#define INCOPIABLE_H

class Incopiable
{
Incopiable (const Incopiable&);
void operator=(const Incopiable&);
protected:
	Incopiable(){}
	~Incopiable(){}
};

#endif