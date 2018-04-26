// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "mainwindow.h"


MainWindow *mainWindow = NULL;


int main(int argc, char **argv) {
    QApplication::setOrganizationName("jkyprian");
    QApplication::setApplicationName("akf");
    QApplication app(argc, argv);
    QErrorMessage::qtHandler()->setMinimumWidth(600);
    QSettings settings;

    #ifdef Q_WS_WIN
    if (QSysInfo::windowsVersion() >= QSysInfo::WV_VISTA) {
        if (settings.value("wddmCheck", true).toBool()) {
            int result = QMessageBox::warning(NULL, "Warning",
                "You are running Windows Vista or later. This program performs extensive "
                "processing on the GPU. You may have to adjust your "
                "<a href='http://www.microsoft.com/whdc/device/display/wddm_timeout.mspx'>WDDM timeout</a> "
                "settings. Do you want to continue?",
                QMessageBox::Yes | QMessageBox::No
            );
            if (result != QMessageBox::Yes) 
                exit(1);
        }
        settings.setValue("wddmCheck", false);
    }
    #endif

 	QString defaultFile = settings.value("lastFile").toString();
	if (defaultFile.isEmpty()) {
		QFileInfo fi(":/test_512x512.png");
		defaultFile = fi.absoluteFilePath();
	}

	mainWindow = new MainWindow;
    mainWindow->restoreGeometry(settings.value("mw").toByteArray());
    mainWindow->showNormal();
	if (QFile::exists(defaultFile)) mainWindow->open(defaultFile);

    app.connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));
	int result = app.exec();

	settings.setValue("mw", mainWindow->saveGeometry());
    delete mainWindow;
    mainWindow = NULL;
	
    return result;
}
