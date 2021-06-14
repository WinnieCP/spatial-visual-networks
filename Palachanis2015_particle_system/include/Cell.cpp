#include "Cell.h"

Cell* current;
Cell* neighbor;

// Class Declarations

// Default Constructor
Cell::Cell()
{
//    Area = 0.025*M_PI;
    prevVelX = 0;
    prevVelY = 0;
    length = 0.5;
    width = 0.05;
	theta = 2*M_PI*((double)rand()/(double)RAND_MAX);
	ID = unique_ID();
	IDNr =0;
}

void Cell::setIDNr(int ID)
{
	IDNr=ID;
}

int Cell::getIDNr()
{
	return IDNr;
}

void Cell::InitPos(int RANGE)
{
    int BASE = -(RANGE/2);
        initX = rand() % RANGE + BASE;
        prevX = initX;
        currX = initX;
        initY = rand() % RANGE + BASE;
    prevY = initY;
        currY = initY;
        RotX = 0;
        RotY = 0;
        setVelX(0);
        setVelY(0);
}

void Cell::InitPos2(double x, double y)
{
	initX = x;
	prevX = initX;
	currX = initX;
	initY = y;
    prevY = initY;
	currY = initY;
	RotX = 0;
	RotY = 0;
	setVelX(0);
	setVelY(0);
}

void Cell::InitOrient(double hx, double hy)
{
	theta= atan2(hy,hx);
}

void Cell::setInitPos(double inputX, double inputY)
{
	initX = inputX;
	prevX = initX;
	currX = initX;
	initY = inputY;
	prevY = initY;
	currY = initY;
	RotX = 0;
	RotY = 0;
}
void Cell::shiftPos(double ll, double bl)
{
	initX = initX - bl*ll/2.*cos(theta);
	initY = initY - bl*ll/2.*sin(theta);
	currX = initX;
	currY = initY;
	prevX = initX;
	prevY = initY;
}

unsigned long long int Cell::unique_ID()
{
	static unsigned long long int ID = 0;
	return ++ID;
}

int Cell::getID()
{
	return ID;
}

double Cell::getWidth()
{
	return width;
}

void Cell::setWidth(double input_w, double input_l)
{
    width = input_w*input_l/2.;
}

double Cell::getLength()
{
    return length;
}

void Cell::setLength(double input)
{
    length = input/2.;
}

// Position methods
double Cell::getInitX()
{
	return initX;
}

double Cell::getInitY()
{
	return initY;
}

double Cell::getCurrX()
{
	return currX;
}

void Cell::setCurrX(double input)
{
	currX = input;
}

double Cell::getCurrY()
{
	return currY;
}

void Cell::setCurrY(double input)
{
	currY = input;
}

double Cell::getPrevX()
{
	return prevX;
}

void Cell::setPrevX(double input)
{
	prevX = input;
}

double Cell::getPrevY()
{
	return prevY;
}

void Cell::setPrevY(double input)
{
	prevY = input;
}

double Cell::getRotX()
{
	return RotX;
}

void Cell::setRotX(double input)
{
		RotX = input;
}

double Cell::getRotY()
{
	return RotY;
}

void Cell::setRotY(double input)
{
		RotY = input;
}
// End Position methods

// Velocity methods
double Cell::getVelX()
{
	return velX;
}

void Cell::setVelX(double input)
{
	velX = input;
}

double Cell::getVelY()
{
	return velY;
}

void Cell::setVelY(double input)
{
	velY = input;
}

double Cell::getPrevVelX()
{
	return prevVelX;
}

void Cell::setPrevVelX(double input)
{
	prevVelX = input;
}

double Cell::getPrevVelY()
{
	return prevVelY;
}

void Cell::setPrevVelY(double input)
{
	prevVelY = input;
}
// End Velocity methods

// Angle Methods
// Overloading method. One instance is the simple setting of angle.
//The second is the turning of a cell due to another cell in close proximity.
void Cell::setTheta(double input)
{
	theta = input;
}

double Cell::getTheta()
{
	return theta;
}
// End Angle Methods

// Force Methods
double Cell::getSumF()
{
	return sumF;
}

void Cell::setSumF(double input)
{
	sumF = input;
}
// End Force Methods
