#include "SceneXMLDocument.h"

#include "Common.h"

#include "Scene/Camera.h"
#include "Scene/ObjectSequence.h"
#include "Scene/ObjectSequenceInstance.h"
#include "Scene/Scene.h"
#include "Scene/SpotLight.h"
#include "Qt/Settings.h"
#include "Utils/CoutMethods.h"
#include "SfmlView.h"
#include "Lighting/EnvMap.h"

SceneData* SceneXMLDocument::getDataFromFile(QString filename)
{

   QDomDocument doc(QString(filename.toStdString().substr(filename.toStdString().find_last_of("/\\")+1).c_str()));
   QFile file(filename);
   if (!file.open(QIODevice::ReadOnly))
   {
      QMessageBox::warning(0, "Failed Loading XML", QString("Failed to open '%1'").arg(filename));
      return 0;
   }
   if (!doc.setContent(&file))
   {
      QMessageBox::warning(0, "Failed Loading XML", QString("Failed to read '%1'").arg(filename));
      file.close();
      return 0;
   }
   file.close();


   SceneData* data = new SceneData();

   // Default values
   data->windowWidth = 768;
   data->windowHeight = 512;
   data->parameterFilename = ""; // None
   data->automaticRotation = 0;
   
   // the outermost element (Scene)
   QDomElement root = doc.documentElement();
   data->name = root.attribute("Name").toStdString();

   // visit all elements that are direct children of "Scene"
   QDomNode n = root.firstChild();

   while(!n.isNull())
   {
      // try to convert the node to an element.
      QDomElement e = n.toElement(); 
      
      if(!e.isNull()) // the node really is an element.
      {
         // cout << qPrintable(e.tagName()) << endl; 

         if(e.tagName() == "Window")
         {
            // Resolution
            data->windowWidth  = e.attribute("width").toInt();
            data->windowHeight = e.attribute("height").toInt();
            if(data->windowWidth == 800 && data->windowHeight == 600)
            {
               data->windowWidth = 768;
               data->windowHeight = 512;
            }
         }
         if(e.tagName() == "Parameters")
         {
            data->parameterFilename  = e.attribute("File").toStdString();
         }
         else if(e.tagName() == "Camera")
         {
            data->cameraData.fovh   = e.attribute("fov").toFloat();
            data->cameraData.aspect = e.attribute("aspect").toFloat();
            if(abs(float(data->windowWidth) / data->windowHeight - data->cameraData.aspect) > 0.001)
               data->cameraData.aspect = float(data->windowWidth) / data->windowHeight;
            data->cameraData.zNear  = e.attribute("zNear").toFloat();
            data->cameraData.zFar   = e.attribute("zFar").toFloat();
            data->cameraData.currentPoseIndex = 0; // default

            QDomElement p = e.firstChildElement("Pose");
            
            // Get Poses
            while(!p.isNull())
            {
               Pose pose;

               // Get Eye 
               QDomElement child = p.firstChildElement("Eye");
               pose.userPosition.x = child.attribute("x").toFloat();
               pose.userPosition.y = child.attribute("y").toFloat();
               pose.userPosition.z = child.attribute("z").toFloat();

               // Get View Direction
               child = p.firstChildElement("ViewDir");
               pose.angleX = child.attribute("angleX").toFloat();
               pose.angleY = child.attribute("angleY").toFloat();
               pose.angleZ = child.attribute("angleZ").toFloat();
               pose.viewDirection.x = child.attribute("x").toFloat();
               pose.viewDirection.y = child.attribute("y").toFloat();
               pose.viewDirection.z = child.attribute("z").toFloat();

               // Get Up Vector
               child = p.firstChildElement("Up");
               pose.upVector.x = child.attribute("x").toFloat();
               pose.upVector.y = child.attribute("y").toFloat();
               pose.upVector.z = child.attribute("z").toFloat();

               data->cameraData.poses.push_back(pose);
               p = p.nextSiblingElement("Pose"); // next pose
            }

            p = e.firstChildElement("CurrentPose");
            if(!p.isNull())
            {
               data->cameraData.currentPoseIndex = p.attribute("index").toFloat();
            }

         }
         else if(e.tagName() == "Lights")
         {
            // Get Spot Lights
            QDomElement light = e.firstChildElement("SpotLight");
            while(!light.isNull())
            {
               SpotLightData spot;
               spot.constantAttenuation   = light.attribute("constantAttenuation").toFloat();
               spot.quadraticAttenuation  = light.attribute("quadraticAttenuation").toFloat();
               spot.angleX                = light.attribute("angleX").toFloat();
               spot.angleY                = light.attribute("angleY").toFloat();
               spot.angleZ                = light.attribute("angleZ").toFloat();
               spot.cutoffAngle           = light.attribute("cutoffAngle").toFloat();
               spot.spotExponent          = light.attribute("spotExponent").toFloat();

               // Get Position and I
               QDomElement child = light.firstChildElement("Position");
               spot.position.x = child.attribute("x").toFloat();
               spot.position.y = child.attribute("y").toFloat();
               spot.position.z = child.attribute("z").toFloat();

               child = light.firstChildElement("I");
               spot.I.x = child.attribute("r").toFloat();
               spot.I.y = child.attribute("g").toFloat();
               spot.I.z = child.attribute("b").toFloat();

               data->spotLights.push_back(spot);
               light = light.nextSiblingElement("SpotLight"); // get next spot light
            }
         }
         else if(e.tagName() == "StaticElement")
         {
            StaticElementData stat;
            parseCommonElement(e, stat);

             // Get Position and Rotation
            
            // Position
            QDomElement child = e.firstChildElement("Position");
            stat.position.x = child.attribute("x").toFloat();
            stat.position.y = child.attribute("y").toFloat();
            stat.position.z = child.attribute("z").toFloat();

            // Rotation
            child = e.firstChildElement("Rotation");
            stat.rotation.x = child.attribute("angleX").toFloat();
            stat.rotation.y = child.attribute("angleY").toFloat();
            stat.rotation.z = child.attribute("angleZ").toFloat();

            // Scale
            child = e.firstChildElement("Scale");
            stat.scaleFactor = child.isNull() ? 1.0 : child.attribute("factor").toFloat();


            data->staticElements.push_back(stat);

         }
         else if(e.tagName() == "DynamicElement")
         {
            DynamicElementData dyn;

            parseDynamicElement(e, dyn);

            data->dynamicElements.push_back(dyn);
         }
         else if(e.tagName() == "DynamicElements")
         {
            if(e.hasAttribute("automaticRotation"))
               data->automaticRotation = static_cast<bool>(e.attribute("automaticRotation").toInt());

         }
      }

      n = n.nextSibling();
   }

   // coutSceneData(*data);

   // system("pause");

   return data;
}


void SceneXMLDocument::parseCommonElement(QDomElement& element, CommonElementData& commData)
{
   commData.pathModel    = element.attribute("pathModel").toStdString();
   commData.name         = element.attribute("name").toStdString();

   // Get Atlas and LoaderSettings
   QDomElement child     = element.firstChildElement("Atlas");
   commData.pathAtlas    = child.attribute("pathAtlas").toStdString();
   commData.atlasWidth   = child.attribute("atlasWidth").toInt();
   commData.atlasHeight  = child.attribute("atlasHeight").toInt();
   if(commData.atlasWidth <= 0)
      commData.atlasWidth = 128;
   if(commData.atlasHeight <= 0)
      commData.atlasHeight = 128;

   child = element.firstChildElement("LoaderSettings");
   commData.defaultDrawMode = child.attribute("defaultDrawMode").toUInt();
   commData.unitized        = static_cast<bool>(child.attribute("unitized").toInt());
   commData.centered        = static_cast<bool>(child.attribute("centered").toInt());
   commData.fixedScaleFactor     = child.attribute("scaleFactor").toFloat();
   commData.computedVertexNormals = static_cast<bool>(child.attribute("computedVertexNormals").toInt());
   commData.vertexNormalsAngle    = child.attribute("vertexNormalsAngle").toFloat();
   commData.vertexNormalsSmoothingGroups = static_cast<bool>(child.attribute("vertexNormalsSmoothingGroups").toInt());

}

void SceneXMLDocument::parseDynamicElement(QDomElement& element,
                                           DynamicElementData& dyn)
{
   dyn.animFileStartIndex = 0;
   dyn.animFileEndIndex = 0;
   dyn.pathSequence = "";
   dyn.sequenceReadInMethod = -1;

   parseCommonElement(element, dyn);

   // Get Instances
   QDomElement instance = element.firstChildElement("Instance");

   while(!instance.isNull())
   {
      DynamicInstanceData instData;
      instData.isUserMovable  = static_cast<bool>(instance.attribute("isUserMovable").toInt());
      
      // default values for animation, there might be none
      instData.stepInterval = 40;
      instData.looping  = true;
      instData.forwards = true;
      instData.startAtFrame = 0;

      // Position
      QDomElement instanceChild = instance.firstChildElement("Position");
      instData.position.x = instanceChild.attribute("x").toFloat();
      instData.position.y = instanceChild.attribute("y").toFloat();
      instData.position.z = instanceChild.attribute("z").toFloat();
      
      // Rotation
      instanceChild = instance.firstChildElement("Rotation");
      instData.rotation.x = instanceChild.attribute("angleX").toFloat();
      instData.rotation.y = instanceChild.attribute("angleY").toFloat();
      instData.rotation.z = instanceChild.attribute("angleZ").toFloat();

      // Scale
      instanceChild = instance.firstChildElement("Scale");
      instData.scaleFactor = instanceChild.isNull() ? 1.0 : instanceChild.attribute("factor").toFloat();


      // Maybe AnimationSettings
      instanceChild = instance.firstChildElement("AnimationSettings");
      if(!instanceChild.isNull())
      {
         instData.stepInterval       = instanceChild.attribute("stepInterval").toInt();
         instData.looping            = static_cast<bool>(instanceChild.attribute("looping").toInt());
         instData.forwards           = static_cast<bool>(instanceChild.attribute("forwards").toInt());
         instData.startAtFrame       = instanceChild.attribute("startAtFrame").toInt();
      }

      dyn.instances.push_back(instData);
      instance = instance.nextSiblingElement("Instance");

   }

   // Finished parsing instances, maybe there is a <AnimationLoading> Element
   QDomElement animLoad = element.firstChildElement("AnimationLoading");

   if(!animLoad.isNull())
   {
      dyn.animFileStartIndex = animLoad.attribute("animFileStartIndex").toInt();
      dyn.animFileEndIndex   = animLoad.attribute("animFileEndIndex").toInt();

      dyn.pathSequence = animLoad.attribute("pathSequence").toStdString();
      dyn.sequenceReadInMethod = animLoad.attribute("sequenceReadInMethod").toInt();

   }

}


void SceneXMLDocument::saveSceneXML(QFile& file, int windowWidth, int windowHeight, SfmlView* sfmlView)
{
   // open an xml stream writer and write data
   QXmlStreamWriter  stream( &file );
   stream.setAutoFormatting( true );
   stream.writeStartDocument();

   stream.writeStartElement( "Scene" );
   stream.writeAttribute( "Name", QString::fromStdString(SCENE->getName()).toLower() );
   stream.writeAttribute( "Date", QDateTime::currentDateTime().toString(Qt::ISODate) );


   QString stamp = QDate::currentDate().toString("_ddMMyyyy_")+QTime::currentTime().toString("hhmmss");
   QString initialPath = QDir::currentPath() + "/SceneXML/ParameterSets/" + "ParameterSet" + stamp + ".xml";

   QString filename = QFileDialog::getSaveFileName(0, "Save Parameters As XML",
      initialPath,
      QString("%1 Files (*.%2);;All Files (*)")
      .arg("XML")
      .arg("xml"));

   if (!filename.isEmpty() ) 
   {
      // open the file and check we can write to it
      QFile file( filename );
      if ( !file.open( QIODevice::WriteOnly ) )
      {
         QMessageBox::warning(0, "Save Parameter XML", QString("Failed to write to '%1'").arg(filename));
      }
      else
      {
         saveParameterXML(file, sfmlView);

         stream.writeStartElement( "Parameters" );
         stream.writeAttribute( "File", filename);
         stream.writeEndElement();
      }


   }

   // window
   stream.writeComment("\nResolution\n");
   stream.writeStartElement( "Window" );
   stream.writeAttribute( "width", QString::number(windowWidth));
   stream.writeAttribute( "height", QString::number(windowHeight));
   stream.writeEndElement();


   // write camera info
   stream.writeComment("\nCamera Definition\n");
   const Camera* const camera = SCENE->getCamera();
   stream.writeStartElement( "Camera" );
   stream.writeAttribute("aspect", QString::number(camera->getAspect()));
   stream.writeAttribute("fov", QString::number(camera->getFovh()));
   stream.writeAttribute("zNear", QString::number(camera->getFrustum().zNear));
   stream.writeAttribute("zFar", QString::number(camera->getFrustum().zFar));

   //for(unsigned int p = 0; p < SCENE->getCameraPoses().size(); p++)
   {
      Pose currentPose;
      currentPose.userPosition = SCENE->getCamera()->getUserPosition();
      currentPose.viewDirection = SCENE->getCamera()->getViewDirection();
      currentPose.upVector = SCENE->getCamera()->getUpVector();
      currentPose.angleX = SCENE->getCamera()->getAngleX();
      currentPose.angleY = SCENE->getCamera()->getAngleY();
      currentPose.angleZ = SCENE->getCamera()->getAngleZ();

      stream.writeStartElement( "Pose" );

      stream.writeStartElement( "Eye" );
      stream.writeAttributes(xyz(&currentPose.userPosition[0]));
      stream.writeEndElement();  
      stream.writeComment("(Eye = UserPosition)");

      stream.writeStartElement( "ViewDir" );
      stream.writeAttribute("angleX", QString::number(currentPose.angleX));
      stream.writeAttribute("angleY", QString::number(currentPose.angleY));
      stream.writeAttribute("angleZ", QString::number(currentPose.angleZ));
      stream.writeAttributes(xyz(&currentPose.viewDirection[0]));
      stream.writeEndElement();

      stream.writeStartElement( "Up" );
      stream.writeAttributes(xyz(&currentPose.upVector[0]));
      stream.writeEndElement();

      stream.writeEndElement(); // Pose
   }
   stream.writeStartElement( "CurrentPose" );
   stream.writeAttribute("index", QString::number(0)/*QString::number(SCENE->getCurrentCameraPoseIndex())*/);
   stream.writeEndElement(); // CurrentPose

   stream.writeEndElement(); // Camera


   // write lights info
   stream.writeComment("\nAll Lights\n");
   stream.writeStartElement( "Lights" );

   for(unsigned int i = 0; i < SCENE->getNumSpotLights(); i++)
   {
      writeSpotLight(stream, i);
   }
   stream.writeEndElement(); // Lights

   stream.writeComment("\n\nAll Scene Elements (Static and Dynamic Geometry)\n");

   for(unsigned int i = 0; i < SCENE->getSceneElements().size(); i++)
   {
      writeElement(stream, i);
   }
   stream.writeEndElement(); // Scene

   stream.writeEndDocument(); // Closes all remaining open start elements and writes a newline.

   // close the file 
   file.close();
}


void SceneXMLDocument::writeElement(QXmlStreamWriter& stream, int elementIndex)
{
   const ObjectSequence* const e = Scene::Instance()->getSceneElements().at(elementIndex);
   const ObjModel* const m = e->getModel(0);

   bool dynamic = !e->isStatic();
   
   if(dynamic)
      stream.writeStartElement( "DynamicElement" );
   else
      stream.writeStartElement( "StaticElement" );

   stream.writeAttribute("name", QString::fromStdString(e->getName()).toLower());
   stream.writeAttribute("pathModel", QString::fromStdString(m->getPathModel()));

   // Texture Atlas Information
   stream.writeStartElement( "Atlas" );
   stream.writeAttribute("pathAtlas", QString::fromStdString(m->getPathAtlas()));
   stream.writeAttribute("atlasWidth", QString::number(e->getAtlasWidth()));
   stream.writeAttribute("atlasHeight", QString::number(e->getAtlasHeight()));
   stream.writeEndElement();

   // OBJ Loader and Initial Geometry Processin Information
   stream.writeComment(QString("mode = ") + QString::fromStdString(ObjModel::drawModeToString(e->getDefaultDrawMode())));
   stream.writeStartElement( "LoaderSettings" );
   stream.writeAttribute("defaultDrawMode", QString::number(e->getDefaultDrawMode()));
   stream.writeAttribute("computedVertexNormals", QString::number(static_cast<int>(m->hasComputedVertexNormals())));
   stream.writeAttribute("vertexNormalsAngle", QString::number(m->getVertexNormalsAngle()));
   stream.writeAttribute("vertexNormalsSmoothingGroups", QString::number(static_cast<int>(m->usesVertexNormalsSmoothingGroups())));
   stream.writeAttribute("unitized", QString::number(static_cast<int>(m->isUnitized())));
   stream.writeAttribute("centered", QString::number(static_cast<int>(m->isCentered())));
   stream.writeAttribute("scaleFactor", QString::number(m->getScaleFactor()));
   stream.writeEndElement();

   if(!dynamic) // Position and Rotation for static element
   {
      stream.writeStartElement( "Position" );
      stream.writeAttributes(xyz(&e->getInstance(0)->getPosition()[0]));
      stream.writeEndElement();
      stream.writeStartElement( "Rotation" );
      stream.writeAttribute("angleX", QString::number(e->getInstance(0)->getRotation().x));
      stream.writeAttribute("angleY", QString::number(e->getInstance(0)->getRotation().y));
      stream.writeAttribute("angleZ", QString::number(e->getInstance(0)->getRotation().z));
      stream.writeEndElement();
      stream.writeStartElement( "Scale" );
      stream.writeAttribute("factor", QString::number(e->getInstance(0)->getScaleFactor()));
      stream.writeEndElement();
   }
   else 
   {
      // Write all instances
      for(unsigned int i = 0; i < e->getNumInstances(); i++)
      {
         stream.writeStartElement( "Instance" );
         stream.writeAttribute("isUserMovable", QString::number(static_cast<int>(e->getInstance(i)->isUserMovable())));

         // Position and Rotation
         stream.writeStartElement( "Position" );
         stream.writeAttributes(xyz(&e->getInstance(i)->getPosition()[0]));
         stream.writeEndElement();
         stream.writeStartElement( "Rotation" );
         stream.writeAttribute("angleX", QString::number(e->getInstance(i)->getRotation().x));
         stream.writeAttribute("angleY", QString::number(e->getInstance(i)->getRotation().y));
         stream.writeAttribute("angleZ", QString::number(e->getInstance(i)->getRotation().z));
         stream.writeEndElement();
         stream.writeStartElement( "Scale" );
         stream.writeAttribute("factor", QString::number(e->getInstance(i)->getScaleFactor()));
         stream.writeEndElement();


         // AnimationSettings
         if(e->getLoadedModelCount() > 1)
         {
            stream.writeStartElement( "AnimationSettings" );
            stream.writeAttribute("looping", QString::number(e->getInstance(i)->isLooping())); 
            stream.writeAttribute("forwards", QString::number(e->getInstance(i)->playingForwards())); 
            stream.writeAttribute("stepInterval", QString::number(e->getInstance(i)->getStepInterval())); 
            stream.writeAttribute("startAtFrame", QString::number(e->getInstance(i)->getCurrentFrameIndex())); 
            stream.writeEndElement();
         }

         stream.writeEndElement(); // Instance


      }


      if(e->getLoadedModelCount() > 1)
      {
         stream.writeStartElement( "AnimationLoading" );
         stream.writeAttribute("pathSequence", QString::fromStdString(e->getPathSequence()));
         stream.writeAttribute("sequenceReadInMethod", QString::number(e->getSequenceReadInMethod()));
         stream.writeAttribute("animFileStartIndex", QString::number(e->getAnimFileStartIndex())); 
         stream.writeAttribute("animFileEndIndex", QString::number(e->getAnimFileEndIndex())); 
         stream.writeAttribute("numAnimFiles", QString::number(e->getLoadedModelCount())); // more than 1 means: animation with obj-file sequence
         stream.writeEndElement();
      }
   }


   stream.writeEndElement(); // Element

}

void SceneXMLDocument::writeSpotLight(QXmlStreamWriter& stream, int spotIndex)
{
   const SpotLight* const spot = Scene::Instance()->getSpotLights().at(spotIndex);
   stream.writeStartElement( "SpotLight" );

   stream.writeAttribute("constantAttenuation", QString::number(spot->getConstantAttenuation()));
   stream.writeAttribute("quadraticAttenuation", QString::number(spot->getQuadraticAttenuation()));
   stream.writeAttribute("cutoffAngle", QString::number(spot->getCutoffAngle()));
   stream.writeAttribute("spotExponent", QString::number(spot->getExponent()));
   stream.writeAttribute("angleX", QString::number(spot->getAngleX()));
   stream.writeAttribute("angleY", QString::number(spot->getAngleY()));
   stream.writeAttribute("angleZ", QString::number(spot->getAngleZ()));
   stream.writeStartElement( "Position" );
   stream.writeAttributes(xyz(&spot->getPosition()[0]));
   stream.writeEndElement();
   stream.writeStartElement( "I" );
   stream.writeAttributes(rgb(&spot->getI()[0]));
   stream.writeEndElement();

   stream.writeStartElement( "SpotDirection" );
   stream.writeAttributes(xyz(&spot->getSpotDirection()[0]));
   stream.writeEndElement();

   stream.writeStartElement( "Up" );
   stream.writeAttributes(xyz(&spot->getUpVector()[0]));
   stream.writeEndElement();

   stream.writeEndElement(); // SpotLight
}

QXmlStreamAttributes SceneXMLDocument::rgb(float r, float g, float b)
{
   QXmlStreamAttributes a;
   a.append("r", QString::number(r));
   a.append("g", QString::number(g));
   a.append("b", QString::number(b));

   return a;
}

QXmlStreamAttributes SceneXMLDocument::rgb(const GLfloat* const values)
{
   return rgb(values[0], values[1], values[2]);
}

QXmlStreamAttributes SceneXMLDocument::xyz(float x, float y, float z)
{
   QXmlStreamAttributes a;
   a.append("x", QString::number(x));
   a.append("y", QString::number(y));
   a.append("z", QString::number(z));

   return a;
}

QXmlStreamAttributes SceneXMLDocument::xyz(const GLfloat* const values)
{
   return xyz(values[0], values[1], values[2]);
}



// ----------------------- Parameter XML -----------------------------//

void SceneXMLDocument::saveParameterXML(QFile& file, SfmlView* sfmlView)
{
   // open an xml stream writer and write data
   QXmlStreamWriter  stream( &file );
   stream.setAutoFormatting( true );
   stream.writeStartDocument();

   stream.writeStartElement( "Parameters" );
   stream.writeAttribute( "Date", QDateTime::currentDateTime().toString(Qt::ISODate) );

   stream.writeStartElement( "IndirectLight" );
   stream.writeAttribute("BufferSize", QString::number(SETTINGS->getCurrentILBufferSize()));
   int mode = 2;
   if(SETTINGS->indirectLight_E_ind_Enabled())
      mode = 0;
   else if(SETTINGS->indirectLight_L_ind_Enabled())
      mode = 1;
   stream.writeAttribute("Mode", QString::number(mode));
   stream.writeAttribute("EnvMapRotation", QString::number(SCENE->getEnvMap()->getRotationAngle()));

   stream.writeStartElement( "General" );
   stream.writeAttribute("Rays", QString::number(SETTINGS->getNumRays()));
   stream.writeAttribute("Steps", QString::number(SETTINGS->getNumSteps()));
   stream.writeAttribute("VoxelResolution", QString::number(SETTINGS->getVoxelTextureResolution()));
   stream.writeAttribute("Radius", QString::number(SETTINGS->getRadius()));
   stream.writeAttribute("Spread", QString::number(SETTINGS->getSpread()));
   stream.writeAttribute("PatternSize", QString::number(SETTINGS->getRandomPatternSize()));
   stream.writeEndElement(); // General

   stream.writeStartElement( "Brightness" );
   stream.writeAttribute("IndirectLightScale", QString::number(SETTINGS->getIndirectLightScaleFactor()));
   stream.writeAttribute("DirectLightScale", QString::number(SETTINGS->getDirectLightScaleFactor()));
   stream.writeAttribute("EnvMapOcclusionStrength", QString::number(SETTINGS->getOcclusionStrength()));
   stream.writeAttribute("EnvMapBrightness", QString::number(SETTINGS->getEnvMapBrightness()));
   stream.writeEndElement(); // Brightness

   stream.writeStartElement( "Offsets" );
   stream.writeAttribute("RayOffsetScale", QString::number(SETTINGS->getVoxelOffsetCosThetaScale()));
   stream.writeAttribute("NormalOffsetScale", QString::number(SETTINGS->getVoxelOffsetNormalScale()));
   stream.writeAttribute("DistanceThreshold", QString::number(SETTINGS->getDistanceThresholdScale()));
   stream.writeEndElement(); // Offsets

   stream.writeEndElement(); // Indirect Light

   stream.writeStartElement( "Filter" );
   stream.writeAttribute("enabled", QString::number(SETTINGS->filterEnabled()));

   stream.writeStartElement("Settings");
   stream.writeAttribute("Radius", QString::number(SETTINGS->getFilterRadius()));
   stream.writeAttribute("IterationRadius", QString::number(SETTINGS->getFilterIterationRadius()));
   stream.writeAttribute("Iterations", QString::number(SETTINGS->getFilterIterations()));
   stream.writeAttribute("DistanceLimit", QString::number(SETTINGS->getFilterDistanceLimit()));
   stream.writeAttribute("NormalLimit", QString::number(SETTINGS->getFilterNormalLimitAngle()));
   stream.writeAttribute("MaterialLimit", QString::number(SETTINGS->getFilterMaterialLimit()));
   stream.writeEndElement(); // Settings

   stream.writeStartElement("SurfaceDetail");
   stream.writeAttribute("enabled", QString::number(SETTINGS->surfaceDetailEnabled()));
   stream.writeAttribute("alpha", QString::number(SETTINGS->getSurfaceDetailAlpha()));
   stream.writeEndElement(); // SurfaceDetail

   stream.writeEndElement(); // Filter

   stream.writeStartElement( "ToneMapping" );
   stream.writeAttribute("enabled", QString::number(SETTINGS->toneMappingEnabled()));
   stream.writeAttribute("Linear", QString::number(SETTINGS->linearToneMappingEnabled()));
   stream.writeAttribute("Log", QString::number(SETTINGS->logToneMappingEnabled()));
   stream.writeAttribute("MaxRadiance", QString::number(SETTINGS->getSimpleMaxRadiance()));
   stream.writeEndElement(); // ToneMapping


   stream.writeEndElement(); // Parameters

   stream.writeEndDocument(); // Closes all remaining open start elements and writes a newline.

   // close the file 
   file.close();
}


void SceneXMLDocument::loadParameterXML(QString filename, SfmlView* sfmlView)
{
   QDomDocument doc(QString(filename.toStdString().substr(filename.toStdString().find_last_of("/\\")+1).c_str()));
   QFile file(filename);
   if (!file.open(QIODevice::ReadOnly))
   {
      QMessageBox::warning(0, "Failed Loading XML", QString("Failed to open '%1'").arg(filename));
      return;
   }
   if (!doc.setContent(&file))
   {
      QMessageBox::warning(0, "Failed Loading XML", QString("Failed to read '%1'").arg(filename));
      file.close();
      return;
   }
   file.close();


   // the outermost element 
   QDomElement root = doc.documentElement();

   // visit all elements that are direct children of "Parameters"
   QDomNode n = root.firstChild();

   while(!n.isNull())
   {
      // try to convert the node to an element.
      QDomElement e = n.toElement(); 
      
      if(!e.isNull()) // the node really is an element.
      {
         //cout << qPrintable(e.tagName()) << endl; 

         if(e.tagName() == "IndirectLight")
         {
            SETTINGS->setCurrentILBufferSize(e.attribute("BufferSize").toInt());
            SCENE->getEnvMap()->setRotationAngle(e.attribute("EnvMapRotation", 
               QString::number(SCENE->getEnvMap()->getRotationAngle())).toFloat());
            if(e.hasAttribute("Mode"))
            {
               int mode = e.attribute("Mode").toInt();
               SETTINGS->toggleIndirectLight_E_ind_Enabled(false);
               SETTINGS->toggleIndirectLight_L_ind_Enabled(false);
               SETTINGS->toggleIndirectLightCombinationEnabled(false);
               switch(mode)
               {
               case 0:
                  SETTINGS->toggleIndirectLight_E_ind_Enabled(true);
                  break;
               case 1:
                  SETTINGS->toggleIndirectLight_L_ind_Enabled(true);
                  break;
               case 2:
               default:
                  SETTINGS->toggleIndirectLightCombinationEnabled(true);
                  break;
               }
            }

            QDomElement child = e.firstChildElement("General");
            if(!child.isNull())
            {
               SETTINGS->setNumRays(child.attribute("Rays").toInt());
               SETTINGS->setNumSteps(child.attribute("Steps").toInt());
               SETTINGS->setVoxelTextureResolution(child.attribute("VoxelResolution").toInt());
               SETTINGS->setRadius(child.attribute("Radius").toFloat());
               SETTINGS->setSpread(child.attribute("Spread").toFloat());
               SETTINGS->setRandomPatternSize(child.attribute("PatternSize").toInt());
            }

            child = child.nextSiblingElement("Brightness");
            if(!child.isNull())
            {
               SETTINGS->setIndirectLightScaleFactor(child.attribute("IndirectLightScale").toFloat());
               SETTINGS->setDirectLightScaleFactor(child.attribute("DirectLightScale").toFloat());
               SETTINGS->setEnvMapBrightness(child.attribute("EnvMapBrightness").toFloat());
               SETTINGS->setOcclusionStrength(child.attribute("EnvMapOcclusionStrength").toFloat());
            }

            child = child.nextSiblingElement("Offsets");
            if(!child.isNull())
            {
               SETTINGS->setVoxelOffsetCosThetaScale(child.attribute("RayOffsetScale").toFloat());
               SETTINGS->setVoxelOffsetNormalScale(child.attribute("NormalOffsetScale").toFloat());
               SETTINGS->setDistanceThresholdScale(child.attribute("DistanceThreshold").toFloat());
            }
         }
         else if(e.tagName() == "Filter")
         {
            SETTINGS->toggleFilterEnabled(static_cast<bool>(e.attribute("enabled").toInt()));

            QDomElement child = e.firstChildElement("Settings");
            if(!child.isNull())
            {
               SETTINGS->setFilterRadius(child.attribute("Radius").toInt());
               SETTINGS->setFilterIterations(child.attribute("Iterations").toInt());
               SETTINGS->setFilterIterationRadius(child.attribute("IterationRadius").toInt());
               SETTINGS->setFilterDistanceLimit(child.attribute("DistanceLimit").toFloat());
               SETTINGS->setFilterMaterialLimit(child.attribute("MaterialLimit").toFloat());
               SETTINGS->setFilterNormalLimit(child.attribute("NormalLimit").toFloat());
            }
            child = child.nextSiblingElement("SurfaceDetail");
            if(!child.isNull())
            {
               SETTINGS->toggleSurfaceDetailEnabled(static_cast<bool>(child.attribute("enabled").toInt()));
               SETTINGS->setSurfaceDetailAlpha(child.attribute("alpha").toFloat());
            }
         }
         else if(e.tagName() == "ToneMapping")
         {
            SETTINGS->toggleToneMappingEnabled(static_cast<bool>(e.attribute("enabled").toInt()));
            SETTINGS->toggleLinearToneMappingEnabled(static_cast<bool>(e.attribute("Linear").toInt()));
            SETTINGS->toggleLogToneMappingEnabled(static_cast<bool>(e.attribute("Log").toInt()));
            SETTINGS->setSimpleMaxRadiance(e.attribute("MaxRadiance").toFloat());
         }
      }
      n = n.nextSibling();
   }

}