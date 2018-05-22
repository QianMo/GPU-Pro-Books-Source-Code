#ifndef SCENEXMLDOCUMENT_H
#define SCENEXMLDOCUMENT_H

#include <QtGui>
#include <QDomDocument>
#include <QXmlStreamWriter>

#include <iostream>

#include "OpenGL.h"
#include "Scene/SceneDataStructs.h"

using namespace std;

class SfmlView;

class SceneXMLDocument 
{
public:
   SceneXMLDocument() {};
   static SceneData* getDataFromFile(QString filename);

   static void saveSceneXML(QFile& file, int windowWidth, int windowHeight, SfmlView* sfmlView);

   static void saveParameterXML(QFile& file, SfmlView* sfmlView);
   static void loadParameterXML(QString filename, SfmlView* sfmlView);

private:
   // reading
   static void parseCommonElement(QDomElement& element, CommonElementData& commData);
   static void parseDynamicElement(QDomElement& element, DynamicElementData& dyn);

   // writing
   // XML methods
   static QXmlStreamAttributes rgb(float r, float g, float b);
   static QXmlStreamAttributes xyz(float x, float y, float z);
   static QXmlStreamAttributes rgb(const GLfloat* const values);
   static QXmlStreamAttributes xyz(const GLfloat* const values);
   static void writeSpotLight(QXmlStreamWriter& stream, int spotIndex);
   static void writeElement(QXmlStreamWriter& stream, int elementIndex);

};

#endif
