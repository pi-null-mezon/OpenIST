QT += qml quick

CONFIG += c++11

SOURCES += main.cpp \
    qclasslabel.cpp \
    qimagedirectory.cpp \
    qlabelfileparser.cpp

HEADERS += \
    qclasslabel.h \
    qimagedirectory.h \
    qlabelfileparser.h

RESOURCES += qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
# QML_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


