QT += core
QT -= gui

CONFIG += c++11

TARGET = DataSegment
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

win32 {
    OPENIST_PATH = C:/Programming/OpenIST
}
linux {
    OPENIST_PATH = /home/alex/Programming/openist
}

include($${OPENIST_PATH}/Sharedfiles/opencv.pri)
include($${OPENIST_PATH}/Sharedfiles/openmp.pri)
include($${OPENIST_PATH}/Sharedfiles/tinydnn.pri)

INCLUDEPATH += $${OPENIST_PATH}/CMDUtils/SegNet

SOURCES += $${OPENIST_PATH}/CMDUtils/SegNet/cnnsegmentnet.cpp
