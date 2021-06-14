# Directories
IDIR = ./include
ODIR = ../obj
LDIR = ../usr/include/c++/4.8.2
BDIR = ../build
GSL_LOC = $(HOME)

# Compiler options
CC = g++
LIBS = -L$(GSL_LOC)/lib -lgsl -lgslcblas
INCLUDES = -I $(IDIR) -I$(GSL_LOC)/include/gsl
CFLAGS = -O3 -Wall

# Header files
_DEPS = Cell.h program_constants.h config.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS)))

# Source files
_OBJ = Cell.o ellipse_ellipse_overlap.o main.o particle_mov_angle.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# Executables
EXEC = pS

# Rules
.PHONY: all
all: $(EXEC)

Cell.o: Cell.h Cell.cpp
	g++ -O3 -Wall -c Cell.cpp

ellipse_ellipse_overlap.o: Cell.h config.h program_constants.h ellipse_ellipse_overlap.cpp
	g++ -O3 -Wall -c ellipse_ellipse_overlap.cpp

$(ODIR)/%.o: %.cpp
	$(CC) -c -o $@ $< $(INCLUDES) $(CFLAGS)

pS: $(OBJ)
	$(CC) -o $(BDIR)/$@ $^ $(INCLUDES) $(CFLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*.o *~ core $(BDIR)/*

#ODIR -c -o $@ $<

