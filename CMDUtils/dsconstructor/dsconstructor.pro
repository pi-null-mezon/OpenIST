QT += core
QT -= gui

CONFIG += c++11

TARGET = dsconstructor
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

OPENIST_PATH = C:/Programming/OpenIST

include($${OPENIST_PATH}/Sharedfiles/opencv.pri)
