#ifndef PAGETONEMAPPING_H
#define PAGETONEMAPPING_H

#include <QtGui>

class PageToneMapping: public QWidget
{
   Q_OBJECT

public:
   PageToneMapping(QWidget* parent = 0);


public slots:

signals:

private:

   void addTonemapControl();
   void addMiscControl();

   QVBoxLayout* layout;
};

#endif
