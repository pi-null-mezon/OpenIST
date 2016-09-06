QT += core
QT -= gui

CONFIG += c++11

TARGET = cnnsegment
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
           qimagefinder.cpp \
           cnnsegmentnet.cpp

HEADERS += \
           qimagefinder.h \
           cnnsegmentnet.h

OPENIST_PATH = C:/Programming/OpenIST		   
		   
include( $${OPENIST_PATH}/Sharedfiles/opencv.pri )
include( $${OPENIST_PATH}/Sharedfiles/tinycnn.pri )
include( $${OPENIST_PATH}/Sharedfiles/openmp.pri )
