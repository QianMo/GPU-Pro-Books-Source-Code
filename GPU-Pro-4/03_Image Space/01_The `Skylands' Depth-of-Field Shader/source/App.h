/**
  \file App.h
 */
#ifndef App_h
#define App_h

#include <G3D/G3DAll.h>
#include "VVDoF.h"

class App : public GApp {
    
    shared_ptr<VVDoF>    m_depthOfField;
    VVDoF::DebugOption   m_debugOption;

    GuiDropDownList*     m_sceneDropDownList;
    shared_ptr<Scene>    m_scene;

    /** Used for enabling dragging of objects with m_splineEditor.*/
    shared_ptr<Entity>   m_selectedEntity;

    /** Used for editing entity splines.*/
    PhysicsFrameSplineEditor::Ref   m_splineEditor;

    GuiDropDownList*     m_entityList;

    /** Don't allow object editing */
    bool                 m_preventEntityDrag;
    bool                 m_preventEntitySelect;

    bool                 m_showAxes;
    bool                 m_showLightSources;
    bool                 m_showWireframe;

    /** Loads whatever scene is currently selected in the m_sceneDropDownList. */
    void loadScene();

    /** Save the current scene over the one on disk. */
    void saveScene();

    /** Called from onInit */
    void makeGUI();

    void selectEntity(const Entity::Ref& e);

public:
    
    App(const GApp::Settings& settings = GApp::Settings());

    virtual void onInit() override;
    virtual void onAI() override;
    virtual void onNetwork() override;
    virtual void onSimulation(RealTime rdt, SimTime sdt, SimTime idt) override;
    virtual void onPose(Array<Surface::Ref>& posed3D, Array<Surface2D::Ref>& posed2D) override;

    // You can override onGraphics if you want more control over the rendering loop.
    // virtual void onGraphics(RenderDevice* rd, Array<Surface::Ref>& surface, Array<Surface2D::Ref>& surface2D) override;

    virtual void onGraphics3D(RenderDevice* rd, Array<Surface::Ref>& posed3D) override;
    virtual void onGraphics2D(RenderDevice* rd, Array<Surface2D::Ref>& posed2D) override;

    virtual bool onEvent(const GEvent& e) override;
    virtual void onUserInput(UserInput* ui) override;
    virtual void onCleanup() override;

    void testShader2();

    /** Sets m_endProgram to true. */
    virtual void endProgram();
};

#endif
