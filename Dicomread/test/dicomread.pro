QT += core
QT -= gui

CONFIG += c++11

TARGET = untitled1
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

OPENIST_PATH = C:/Programming/OpenIST

SOURCES += main.cpp \
           ../dicomopencv.cpp

HEADERS += ../dicomopencv.h

INCLUDEPATH += $${OPENIST_PATH}/Dicomread

include( $${OPENIST_PATH}/Sharedfiles/opencv.pri )
include( $${OPENIST_PATH}/Sharedfiles/DCMTK.pri )
