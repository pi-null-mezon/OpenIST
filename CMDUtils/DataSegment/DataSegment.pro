QT += core
QT -= gui

CONFIG += c++11

TARGET = DataSegment
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

OPENIST_PATH = C:/Programming/OpenIST

include($${OPENIST_PATH}/sharedfiles/opencv.pri)
include($${OPENIST_PATH}/sharedfiles/openmp.pri)
include($${OPENIST_PATH}/sharedfiles/tinycnn.pri)

INCLUDEPATH += $${OPENIST_PATH}/CMDUtils/SegNet

SOURCES += $${OPENIST_PATH}/CMDUtils/SegNet/cnnsegmentnet.cpp
