// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#ifndef INCLUDED_MAINWINDOW_H
#define INCLUDED_MAINWINDOW_H

#include "ui_mainwindow.h"
#ifdef HAVE_QUICKTIME
#include "quicktime.h"
#endif

class MainWindow : public QMainWindow, public Ui_MainWindow {
    Q_OBJECT
public:
	MainWindow();
    ~MainWindow();

	bool open(const QString& fileName);

public slots:
    void on_actionOpen_triggered();
    void on_actionSave_triggered();
	void on_actionAbout_triggered();
	void on_algorithm_currentIndexChanged(int index);

    #ifdef HAVE_QUICKTIME
    void on_actionRecord_triggered();
    void on_timeSlider_valueChanged(int value);
	void on_playButton_clicked();
	void do_play();
    #endif

private:
    QString m_fileName;
	int m_algorithm;

    #ifdef HAVE_QUICKTIME
    QTimer *m_timer;
    bool m_haveQuickTime;
    quicktime_player *m_player;
    #endif
};

extern MainWindow *mainWindow;

#endif
