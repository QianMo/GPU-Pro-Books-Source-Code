/** \file App.cpp */
#include "App.h"

// Tells C++ to invoke command-line main() function even on OS X and Win32.
G3D_START_AT_MAIN();

int main(int argc, const char* argv[]) {
    GApp::Settings settings(argc, argv);
    
    // Change the window and other startup parameters by modifying the
    // settings class.  For example:
    settings.window.width       = 1200; 
    settings.window.height      = 900;
    settings.window.defaultIconFilename = "icon.png";

#   ifdef G3D_WIN32
    	if (FileSystem::exists("data-files", false)) {
            // Running on Windows, building inside the starter directory
            chdir("data-files");
        }
#   endif

    return App(settings).run();
}


App::App(const GApp::Settings& settings) : GApp(settings) {
#   ifdef G3D_DEBUG
        // Let the debugger catch unhandled exceptions
        catchCommonExceptions = false;
#   endif

    alwaysAssertM(notNull(m_depthBuffer),  "GApp failed to allocate a depth buffer");
    alwaysAssertM(notNull(m_colorBuffer0), "GApp failed to allocate a color buffer");
}


void App::onInit() {
    GApp::onInit();
    // Called before the application loop begins.  Load data here and
    // not in the constructor so that common exceptions will be
    // automatically caught.

    m_depthOfField        = VVDoF::create();

    showRenderingStats    = false;
    m_showLightSources    = false;
    m_showAxes            = false;
    m_showWireframe       = false;
    m_preventEntityDrag   = false;
    m_preventEntitySelect = false;
    m_debugOption         = VVDoF::NONE;
    setLowerFramerateInBackground(true);

    // For higher-quality screenshots:
    developerWindow->videoRecordDialog->setScreenShotFormat("PNG");
    developerWindow->videoRecordDialog->setCaptureGui(false);

    makeGUI();

    m_scene = Scene::create();
    // Start wherever the developer HUD last marked as "Home"
    defaultCamera->setFrame(bookmark("Home"));

    loadScene();
}


void App::makeGUI() {
    // Turn on the developer HUD
    debugWindow->setVisible(true);
    developerWindow->cameraControlWindow->setVisible(true);
    developerWindow->videoRecordDialog->setEnabled(true);

    GFont::Ref iconFont = GFont::fromFile(System::findDataFile("icon.fnt"));
    
    // Create a scene management GUI
    GuiPane* scenePane = debugPane->addPane("Scene", GuiTheme::ORNATE_PANE_STYLE);
    scenePane->moveBy(0, -10);
    scenePane->beginRow(); {
        // Example of using a callback; you can also listen for events in onEvent or bind controls to data
        m_sceneDropDownList = scenePane->addDropDownList("", Scene::sceneNames(), NULL, GuiControl::Callback(this, &App::loadScene));

        static const char* reloadIcon = "q";
        static const char* diskIcon = "\xcd";

        scenePane->addButton(GuiText(reloadIcon, iconFont, 14), this, &App::loadScene, GuiTheme::TOOL_BUTTON_STYLE)->setWidth(32);
        scenePane->addButton(GuiText(diskIcon, iconFont, 18), this, &App::saveScene, GuiTheme::TOOL_BUTTON_STYLE)->setWidth(32);
    } scenePane->endRow();

    const int w = 120;
    scenePane->beginRow(); {
        scenePane->addCheckBox("Axes", &m_showAxes)->setWidth(w);
        scenePane->addCheckBox("Light sources", &m_showLightSources);
    } scenePane->endRow();
    scenePane->beginRow(); {
        scenePane->addCheckBox("Wireframe", &m_showWireframe)->setWidth(w);
    } scenePane->endRow();
    static const char* lockIcon = "\xcf";
    scenePane->addCheckBox(GuiText(lockIcon, iconFont, 20), &m_preventEntityDrag, GuiTheme::TOOL_CHECK_BOX_STYLE);
    scenePane->pack();

    GuiPane* entityPane = debugPane->addPane("Entity", GuiTheme::ORNATE_PANE_STYLE);
    entityPane->moveRightOf(scenePane);
    entityPane->moveBy(10, 0);
    m_entityList = entityPane->addDropDownList("Name");

    // Dock the spline editor
    m_splineEditor = PhysicsFrameSplineEditor::create("Spline Editor", entityPane);
    addWidget(m_splineEditor);
    developerWindow->cameraControlWindow->moveTo(Point2(window()->width() - developerWindow->cameraControlWindow->rect().width(), 0));
    m_splineEditor->moveTo(developerWindow->cameraControlWindow->rect().x0y0() - Vector2(m_splineEditor->rect().width(), 0));
    entityPane->pack();

    GuiPane* infoPane = debugPane->addPane("Info", GuiTheme::ORNATE_PANE_STYLE);
    infoPane->moveRightOf(entityPane);
    infoPane->moveBy(10, 0);

    // Example of how to add debugging controls
    infoPane->addLabel("You can add more GUI controls");
    infoPane->addLabel("in App::onInit().");
    infoPane->addButton("Exit", this, &App::endProgram);
    infoPane->pack();

    GuiPane* dofPane = debugPane->addPane("Visualize", GuiTheme::ORNATE_PANE_STYLE);
    dofPane->moveRightOf(infoPane);
    dofPane->moveBy(10, 0);

    dofPane->addRadioButton("Output",        VVDoF::NONE, &m_debugOption);
    dofPane->addRadioButton("Signed CoC",    VVDoF::SHOW_SIGNED_COC, &m_debugOption);
    dofPane->addRadioButton("CoC",           VVDoF::SHOW_COC, &m_debugOption);
    dofPane->addRadioButton("Region",        VVDoF::SHOW_REGION, &m_debugOption);
    dofPane->addRadioButton("Blurry Buffer", VVDoF::SHOW_BLURRY, &m_debugOption);
    dofPane->addRadioButton("Mid + Far",     VVDoF::SHOW_MID_AND_FAR, &m_debugOption);
    dofPane->addRadioButton("Near Buffer",   VVDoF::SHOW_NEAR, &m_debugOption);
    dofPane->addRadioButton("Input",         VVDoF::SHOW_INPUT, &m_debugOption);
    dofPane->pack();

    // More examples of debugging GUI controls:
    // debugPane->addCheckBox("Use explicit checking", &explicitCheck);
    // debugPane->addTextBox("Name", &myName);
    // debugPane->addNumberBox("height", &height, "m", GuiTheme::LINEAR_SLIDER, 1.0f, 2.5f);
    // button = debugPane->addButton("Run Simulator");

    debugWindow->pack();
    debugWindow->setRect(Rect2D::xywh(0, 0, window()->width(), debugWindow->rect().height()));
}


void App::loadScene() {
    selectEntity(shared_ptr<Entity>());

    const std::string& sceneName = m_sceneDropDownList->selectedValue().text();

    // Use immediate mode rendering to force a simple message onto the screen
    drawMessage("Loading " + sceneName + "...");

    // Load the scene
    try {
        const Any& any = m_scene->load(sceneName);

        // TODO: Parse extra fields that you've added to the .scn.any
        // file here
        (void)any;

        // Because the CameraControlWindow is hard-coded to the
        // defaultCamera, we have to copy the camera's values here
        // instead of assigning a pointer to it.
        *defaultCamera = *m_scene->defaultCamera();

        defaultController->setFrame(defaultCamera->frame());

        // Populate the Entity list in the GUI
        Array<std::string> nameList;
        m_scene->getEntityNames(nameList);
        m_entityList->clear();
        m_entityList->append("<none>");
        for (int i = 0; i < nameList.size(); ++i) {
            m_entityList->append(nameList[i]);
        }

    } catch (const ParseError& e) {
        const std::string& msg = e.filename + format(":%d(%d): ", e.line, e.character) + e.message;
        drawMessage(msg);
        debugPrintf("%s", msg.c_str());
        System::sleep(5);
        m_scene->clear();
    }
}


void App::onAI() {
    GApp::onAI();
    // Add non-simulation game logic and AI code here
}


void App::onNetwork() {
    GApp::onNetwork();
    // Poll net messages here
}


void App::onSimulation(RealTime rdt, SimTime sdt, SimTime idt) {
    GApp::onSimulation(rdt, sdt, idt);

    m_splineEditor->setEnabled(m_splineEditor->enabled() && ! m_preventEntityDrag);
	
    m_splineEditor->setVisible(m_splineEditor->enabled());

    // Add physical simulation here.  You can make your time
    // advancement based on any of the three arguments.
    if (notNull(m_selectedEntity) && m_splineEditor->enabled()) {
        // Apply the edited spline.  Do this before object simulation, so that the object
        // is in sync with the widget for manipulating it.
        m_selectedEntity->setFrameSpline(m_splineEditor->spline());
    }

    m_scene->onSimulation(sdt);
}


bool App::onEvent(const GEvent& event) {
    if (GApp::onEvent(event)) {
        return true;
    }

    if (event.type == GEventType::VIDEO_RESIZE) {
        // Example GUI dynamic layout code.  Resize the debugWindow to fill
        // the screen horizontally.
        debugWindow->setRect(Rect2D::xywh(0, 0, window()->width(), debugWindow->rect().height()));
    }


    if (! m_preventEntitySelect && (event.type == GEventType::MOUSE_BUTTON_DOWN) && (event.button.button == 0)) {
        // Left click: select by casting a ray through the center of the pixel
        const Ray& ray = defaultCamera->worldRay(event.button.x + 0.5f, event.button.y + 0.5f, renderDevice->viewport());
        
        float distance = finf();

        selectEntity(m_scene->intersect(ray, distance));
    }
    

    if (! m_preventEntitySelect && (event.type == GEventType::GUI_ACTION) && (event.gui.control == m_entityList)) {
        // User clicked on dropdown list
        selectEntity(m_scene->entity(m_entityList->selectedValue().text()));
    }

    // If you need to track individual UI events, manage them here.
    // Return true if you want to prevent other parts of the system
    // from observing this specific event.
    //
    // For example,
    // if ((event.type == GEventType::GUI_ACTION) && (event.gui.control == m_button)) { ... return true;}
    if ((event.type == GEventType::KEY_DOWN) && (event.key.keysym.sym == 'r')) { 
        m_depthOfField->reloadShaders();
        debugPrintf("Reloaded shaders\n");
        return true; 
    }

    return false;
}


void App::selectEntity(const Entity::Ref& e) {
    m_selectedEntity = e;

    if (notNull(m_selectedEntity)) {
        m_splineEditor->setSpline(m_selectedEntity->frameSpline());
        m_splineEditor->setEnabled(! m_preventEntityDrag);
        m_entityList->setSelectedValue(m_selectedEntity->name());
    } else {
        m_splineEditor->setEnabled(false);
    }
}


void App::onUserInput(UserInput* ui) {
    GApp::onUserInput(ui);
    (void)ui;
}


void App::onPose(Array<Surface::Ref>& posed3D, Array<Surface2D::Ref>& posed2D) {
    GApp::onPose(posed3D, posed2D);
    m_scene->onPose(posed3D);
}


void App::onGraphics3D(RenderDevice* rd, Array<Surface::Ref>& allSurfaces) {
     // Cull and sort
    Array<shared_ptr<Surface> > sortedVisibleSurfaces;
    Surface::cull(defaultCamera->frame(), defaultCamera->projection(), rd->viewport(), allSurfaces, sortedVisibleSurfaces);
    Surface::sortBackToFront(sortedVisibleSurfaces, defaultCamera->frame().lookVector());

    // Early-Z pass
    static const bool renderTransmissiveSurfaces = false;
    Surface::renderDepthOnly(rd, sortedVisibleSurfaces, CullFace::BACK, renderTransmissiveSurfaces);

    // Compute AO
    m_ambientOcclusion->update(rd, m_scene->lighting()->ambientOcclusionSettings, defaultCamera, m_depthBuffer);

    // Compute shadow maps and forward-render visible surfaces
    Surface::Environment environment;
    environment.setData(m_scene->lighting(), m_ambientOcclusion, shared_ptr<Texture>(), shared_ptr<Texture>());
    Surface::render(rd, defaultCamera->frame(), defaultCamera->projection(), sortedVisibleSurfaces, allSurfaces, m_scene->lighting(), environment);

    if (m_showWireframe) {
        Surface::renderWireframe(rd, sortedVisibleSurfaces);
    }


    //////////////////////////////////////////////////////
    // Sample immediate-mode rendering code
    if (m_showAxes) {
        Draw::axes(Point3(0, 0, 0), rd);
    }    

    if (m_showLightSources) {
        Draw::lighting(m_scene->lighting(), rd);
    }

    // Call to make the GApp show the output of debugDraw
    drawDebugShapes();

    m_depthOfField->apply(rd, m_colorBuffer0, m_depthBuffer, defaultCamera, m_debugOption);
}


void App::onGraphics2D(RenderDevice* rd, Array<Surface2D::Ref>& posed2D) {
    // Render 2D objects like Widgets.  These do not receive tone mapping or gamma correction
    Surface2D::sortAndRender(rd, posed2D);
}


void App::onCleanup() {
    // Called after the application loop ends.  Place a majority of cleanup code
    // here instead of in the constructor so that exceptions can be caught
}


void App::endProgram() {
    m_endProgram = true;
}


void App::saveScene() {
    // Called when the "save" button is pressed
    Any a = m_scene->toAny();
    const std::string& filename = a.source().filename;
    if (filename != "") {
        a.save(filename);
        debugPrintf("Saved %s\n", filename.c_str());
    } else {
        debugPrintf("Could not save: empty filename");
    }
}
