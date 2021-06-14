#ifndef CELL_H
#define CELL_H
#include <iostream>
#define _USE_MATH_DEFINES
#define EPSILON 0.000001
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
//using namespace std;

class Cell
{
public:
	Cell();
    void InitPos(int RANGE);
	void InitPos2(double x, double y);
	void InitOrient(double hx, double hy);
	void setInitPos(double inputX, double inputY);
	unsigned long long int unique_ID();
	int getID();
	double getInitX();
	double getInitY();
	double getCurrX();
	void setCurrX(double input);
	double getCurrY();
	void setCurrY(double input);
	double getPrevX();
	void setPrevX(double input);
	double getPrevY();
	void setPrevY(double input);
	double getRotX();
	void setRotX(double input);
	double getRotY();
	void setRotY(double input);
	double getVelX();
	void setVelX(double input);
	double getVelY();
	void setVelY(double input);
	double getPrevVelX();
	void setPrevVelX(double input);
	double getPrevVelY();
	void setPrevVelY(double input);
    void setTheta(double input); // set the angle manually
	double getTheta();
	double getLength();
    	void setLength(double input);
	double getWidth();
    	void setWidth(double input_w, double input_l);
	double getSumF();
	void setSumF(double input);
	double radCrit;
	double radEq;
	double radFin;
	void setIDNr(int);
	int getIDNr();
	void shiftPos(double ll, double bl);
private:
	double velX;
	double velY;
	double prevVelX;
	double prevVelY;
	double initX;
	double initY;
	double prevX;
	double prevY;
	double currX;
	double currY;
	double RotX;
	double RotY;
	double length; // length of the long semi-axis
	double width; // length of the short semi-axis
    double Area;
	double theta;
	double sumF;
	int ID;
	int IDNr;
};

//const int RANGE = 500; // range in which particles are initialized
//const int BASE = -250; // start coordinate of space
//const int maxSize = 700; // size of array holding all cells.
//const double dt = 0.1; // 1 simulated minute

typedef Cell* cellPtr;

// ============= Declarations ============= //
// (Winnie) functions to load exp. positions and orientations of ellipses
void LoadAndInitPos(std::string file_path, std::vector<cellPtr> &a);
void LoadAndInitOrient(std::string file_path, std::vector<cellPtr> &a);
// outputs the filename numbered according to the time-step in the format "output_XX.dat"
std::string make_output_filename(size_t index);

// returns the Euclidean distance between 2 cells
double Distance(Cell* current, Cell* next);

// returns the potential between 2 cells based on the Euclidean Distance.
double Potential(Cell* current, Cell* next, double lamda1, double rescaleEllipse, bool debug);

//returns true if two cells overlap
double Overlap(Cell* current, Cell* next);

// Ellipse Ellipse Overlap between 2 cells
double ellipse_ellipse_overlap (double PHI_1, double A1, double B1,
                                double H1, double K1, double PHI_2,
                                double A2, double B2, double H2, double K2,
                                double X[4], double Y[4], int * NROOTS,
                                int *rtnCode);

// stores the "direction" between 2 cells according to the formula : (vector[j]-vector[i])/Distance(j-i). The Cartesian system is the one with center (0,0) .
double eigenX(Cell* current, Cell* next);

// the same as above for the Y component.
double eigenY(Cell* current, Cell* next);

// rotation of cells resembling the Potts Model. Cells are turned by a small random angle. If their energy is minimized,
// the turn is accepted. If the energy is not minimized, then the turn is accepted with Boltzmann probability p = exp(-dE).
void rotateCells(std::vector<cellPtr> cells, double lamda1, double dt);

// calculation of interactions and updating of positions
void move_cells(int time, int initseed,  int maxSize, double ratio, double lamda1, double overdamp, double dt, double bodyl, double distscale, std::string filepath_pos, std::string filepath_orient, std::string outpath, double ll);

//Order Parameter Functions. Zero means no order, 1 means perfect alignment
double cos2Theta(Cell* current, std::vector<cellPtr> particles, double oRange);
double orderParameter(std::vector<cellPtr> cells, double oRange);

// Rounding error because of PI
bool almostEqual(const double x, const double y);
#endif

