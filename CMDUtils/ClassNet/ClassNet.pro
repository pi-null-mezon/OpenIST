QT += core
QT -= gui

CONFIG += c++11

TARGET = ClassNet
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
           qimagefinder.cpp \
           cnnclassnet.cpp

HEADERS += \
           qimagefinder.h \
           cnnclassnet.h

win32 {
    OPENIST_PATH = C:/Programming/OpenIST
}
linux {
    OPENIST_PATH = /home/alex/Programming/openist
}
		   
include( $${OPENIST_PATH}/Sharedfiles/opencv.pri )
include( $${OPENIST_PATH}/Sharedfiles/tinydnn.pri )
include( $${OPENIST_PATH}/Sharedfiles/openmp.pri )
