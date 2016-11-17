#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlcontext>
#include <QQmlEngine>
#include <QVariant>
#include <QList>

#include "qclasslabel.h"
#include "qimagedirectory.h"
#include "qlabelfileparser.h"

int main(int argc, char *argv[])
{       
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.load(QUrl(QLatin1String("qrc:/main.qml")));
    QObject *_pqmlrootobj = engine.rootObjects()[0];

    if(_pqmlrootobj == NULL) {
        qWarning("Can not create qml root object!");
        return -1;
    }

    QList<QClassLabel> _classlist = QLabelFileParser::parseFile("Labels.ini");

    QList<QObject*> _objpointersList;
    for(int i = 0; i < _classlist.size(); i++)
        _objpointersList.append(&_classlist[i]);

    engine.rootContext()->setContextProperty("classList", QVariant::fromValue(_objpointersList));

    QImageDirectory imgDir;
    imgDir.setClassList(&_objpointersList);

    QObject::connect(_pqmlrootobj, SIGNAL(dirUrlChanged(QUrl)), &imgDir, SLOT(setDirUrl(QUrl)));
    QObject::connect(_pqmlrootobj, SIGNAL(askNextFile()), &imgDir, SLOT(getNextFile()));
    QObject::connect(_pqmlrootobj, SIGNAL(askMarksSave()), &imgDir, SLOT(saveFile()));
    QObject::connect(&imgDir, SIGNAL(fileUrl(QVariant)), _pqmlrootobj, SLOT(updateImage(QVariant)));


    return app.exec();
}
