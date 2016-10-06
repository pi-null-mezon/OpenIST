CONFIG += enableopenmp
enableopenmp {
    win32-msvc* {
        QMAKE_CXXFLAGS+= -openmp
    }
    win32-g++ {
        QMAKE_CXXFLAGS+= -fopenmp
        LIBS += -fopenmp
    }
    linux-g++ {
        QMAKE_CXXFLAGS+= -fopenmp
        LIBS += -fopenmp
    }
message(OpenMP enabled)
}

