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

include( opencv.pri )
include( tinycnn.pri )
include( openmp.pri )
