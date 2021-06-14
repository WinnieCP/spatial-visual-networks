#include "Cell.h"
#include "config.h"
#include "program_constants.h"
#include <algorithm>
using namespace std;

#define LONGSIM true
#define GNUPLOT_PIPE false
#define DEBUGGING false

// Function implementations
void LoadAndInitPos(std::string file_path, std::vector<cellPtr> &a, double d)
{
   // printf("Trying to Load position data.\n");
    std::ifstream inputFile( file_path.c_str() );
    unsigned int ii = 0;
    double x = 0;
    double y = 0;
    if (inputFile){
        double value;
        while ( inputFile >> value ) {
            x = d*value;
            if ( inputFile >> value ) y = d*value;
	    a[ii]->InitPos2(x, y);
            ii++;
        }
    }
    else{
	printf("Could not open position file");
    }
}

void LoadAndInitOrient(std::string file_path, std::vector<cellPtr> &a)
{
    std::ifstream inputFile( file_path.c_str() );
    unsigned int ii = 0;
    double hx = 0;
    double hy = 0;
    if (inputFile){
        double value;
        while ( inputFile >> value ) {
            hx = value;
        if ( inputFile >> value ) hy = value;
    	    a[ii]->InitOrient(hx,hy);
            ii++;
        }
    }
    else{
	printf("Could not open orientation file");
    }
}


double Distance(Cell* current, Cell* next) // returns Euclidean distance between 2 cells.
{
	double distX = current->getCurrX() - next->getCurrX();
	double distY = current->getCurrY() - next->getCurrY();
	double dist = fabs(sqrt(distX*distX + distY*distY));
	return dist;
}

double Potential(Cell* current, Cell* next, double lamda1, double rescaleEllipse, bool debug)
{
	//POSITIVE VALUES FOR REPULSION (Lamda1), NEGATIVE FOR ATTRACTION (Lamda2)
	// Changed to no attraction (Winnie)
	double areaRep;
	//double areaRep, areaAtt;
	double distance = Distance(current,next);
	double A1 = rescaleEllipse*current->getLength();
	double B1 = ((rescaleEllipse-1.)*current->getLength())+current->getWidth();
	double theta1 = current->getTheta(); //*180.0/M_PI
	double x1 = current->getCurrX();
	double y1 = current->getCurrY();
	double A2 = rescaleEllipse*next->getLength();
	double B2 = rescaleEllipse*next->getWidth();
	double theta2 = next->getTheta();
	double x2 = next->getCurrX();
	double y2 = next->getCurrY();
	double xrep[4], yrep[4];
	int rtn_rep,  nroots_rep;
	//double xrep[4], yrep[4], xatt[4],yatt[4];
	//int rtn_rep, rtn_att, nroots_rep, nroots_att;
	//double interDist = 2.0*attRange*A1;
	double interDist = 2.0*A1;
	//Poly poly1, poly2, poly3, poly4;


	if(distance <= interDist)
	{
		areaRep = ellipse_ellipse_overlap(theta1, A1, B1, x1, y1,
						  theta2, A2, B2, x2, y2,
						  xrep, yrep, &nroots_rep, &rtn_rep);
		if(debug and areaRep<0)
		{
			printf("Area smaller zero!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
			printf("theta:");
			cout<<theta1;
			printf("\nA1 ");
			cout<<A1;
			printf("\nB1 ");
			cout<<B1;
			printf("\nx1 ");
			cout<<x1;
			printf("\ny1 ");
			cout<<y1;
			printf("\n.....\ntheta2 ");
			cout<<theta2;
			printf("\nA2 ");
			cout<<A2;
			printf("\nB2 ");
			cout<<B2;
			printf("\nx2 ");
			cout<<x2;
			printf("\ny2 ");
			cout<<y2;
			printf("\n##############\n");
		}
		if(rescaleEllipse==1 and (A1!=2.5 or B1!=0.375))
		{
			printf("A1,B1\n");
			cout<<A1;
			printf(" ");
			cout<<B1;
			printf("----\n");
		}
		//areaAtt = ellipse_ellipse_overlap(theta1, attRange*A1, attRange*B1, x1, y1,
		//				  theta2, attRange*A2, attRange*B2, x2, y2,
		//				  xatt, yatt, &nroots_att, &rtn_att);
		//return (lamda1*areaRep - lamda2*areaAtt);
		return (lamda1*areaRep);
	
	}
	else
		return 0.0;
}

double Overlap(Cell* current, Cell* next)
{
	double areaRep;
	double distance = Distance(current,next);
	double A1 = current->getLength();
	double B1 = current->getWidth();
	double theta1 = current->getTheta(); //*180.0/M_PI
	double x1 = current->getCurrX();
	double y1 = current->getCurrY();
	double A2 = next->getLength();
	double B2 = next->getWidth();
	double theta2 = next->getTheta();
	double x2 = next->getCurrX();
	double y2 = next->getCurrY();
	double xrep[4], yrep[4];
	int rtn_rep,  nroots_rep;
	//double xrep[4], yrep[4], xatt[4],yatt[4];
	//int rtn_rep, rtn_att, nroots_rep, nroots_att;
	//double interDist = 2.0*attRange*A1;
	double interDist = 2.0*A1;
	//Poly poly1, poly2, poly3, poly4;


	if(distance <= interDist)
	{
	//	printf("checking overlap of \n");
	//	cout<<current->getIDNr();
	//	printf("\n");
	//	cout<<next->getIDNr();
	//	printf("\n");
		areaRep = ellipse_ellipse_overlap(theta1, A1, B1, x1, y1,
						  theta2, A2, B2, x2, y2,
						  xrep, yrep, &nroots_rep, &rtn_rep);
	//	cout<<areaRep;
	//	printf("\n");
		if(areaRep!=0)
		{
			//printf("Overlap detected!\n");
			return areaRep;
		}
		else
		{
			return 0.;
		}
	}
	else
		return 0.;
}


double eigenX(Cell* current, Cell* next)
{
	double distX = current->getCurrX() - next->getCurrX();
	double dist = Distance(current,next);
	if(dist == 0)
		return 0.0;
	else
	{
		double eigX = distX / dist;
		return eigX;
	}
}

double eigenY(Cell* current, Cell* next)
{
	double distY = current->getCurrY() - next->getCurrY();
	double dist = Distance(current,next);
	if(dist == 0)
		return 0.0;
	else
	{
		double eigY = distY / dist;
		return eigY;
	}
}

void rotateCells(vector<cellPtr> cells, double lamda1, double dt, double rescaleEllipse)
{
// changed to no attraction and no Boltzmann distributed noise on angle (Winnie)
	int maxSize = cells.size();
	int ctr = 0;
	int i = 0;
	double randomvalue = 0;
	double phi = 0;
	double theta1 = 0;
	double deltaE =  0;
	double distance = 0;
	double newPotential = 0;
	double oldPotential = 0;
	double interDist = rescaleEllipse*2.0*cells[0]->getLength();
	
	for (int j=0; j<maxSize; j++)		// loop for N times
	{
		ctr = (int)( (double)(maxSize) * ((double)(rand()))/((double)(RAND_MAX)) );	// pick cell at random
		newPotential = 0;
		oldPotential = cells[ctr]->getSumF();
		deltaE = 0;
		theta1 = cells[ctr]->getTheta();
        	randomvalue = (((double)rand()/(double)RAND_MAX) - 0.5);


		phi = theta1 + sqrt(dt)*M_PI/32*randomvalue; // new angle

		// Test new angle
		cells[ctr]->setTheta(phi);
		for(i=0; i < maxSize; i++) // for all cells
		{
			distance = 0;
			if(i != ctr)
			{
				distance = Distance(cells[ctr],cells[i]);
                		if(distance <= interDist) // only in close proximity
					newPotential += Potential(cells[ctr],cells[i], lamda1, rescaleEllipse, false);
			}
		}
		deltaE = newPotential - oldPotential;
		//cout << "Delta Epsilon cell" << ctr << " " << deltaE << endl;
		if(deltaE < 0.0) // if energy is minimized with the turn, accept it
		{
			cells[ctr]->setTheta(phi);
		}
		else
		{
			cells[ctr]->setTheta(theta1);
		}
	}
}

void move_cells(int time, int initseed, int maxSize, double ratio, double lamda1, double overdamp, double dt, double bodyl, double distscale, std::string filepath_pos, std::string filepath_orient, std::string outpath, double ll)
{
//
// Function controlling whole simulation
//
	//bool DEBUGGING=true;
	if(DEBUGGING)
	{
		printf("\n ***********\n");
		cout<<bodyl;
		printf("  ");
		cout<<ratio;
		printf("\n ***********\n");	
	}	
	double rescale=1.1;
	int noOverlaps = 0;
	int i = 0;
	int tMax = time; // maximum time
	int t = 0; // time
	int j = 0;
	int sumOverlap;
	double sumX = 0;
	double sumY = 0;
	double overlap;
	double potential, sumPotential;
	FILE *output1, *output2, *gnuplotpipe;
	std::string outfilepathpos, outfilepathorient;

	srand(initseed); // give system time as seed to random generator
	int initRange = 16*int(sqrt(maxSize));
	vector<cellPtr> particles(maxSize);// create array of pointers of type Cell.
	// Initialize cells (particles)
	//
	for(i=0; i<maxSize; i++)
	{
		particles[i] = new Cell(); 		// initialize in memory
		particles[i]->InitPos(initRange); 	// give initial positions
		particles[i]->setWidth(ratio, bodyl);
		particles[i]->setLength(bodyl);
		particles[i]->setIDNr(i);
	}
	if(DEBUGGING)
	{
		printf("\n.................\n");
		printf("\nwidth: ");
		cout<<particles[0]->getWidth();
		printf("\nlength: ");
		cout<<particles[0]->getLength();
		printf("\n.................\n");
		printf("successfully initialized");
		std::cout<<filepath_pos;
	}
	LoadAndInitPos(filepath_pos, particles, distscale);
	LoadAndInitOrient(filepath_orient, particles);
	for(i=0; i<maxSize; i++)
	{
	//	std::cout<<particles[i]->getInitX();
		particles[i]->shiftPos(ll,bodyl);
	//	std::cout<<particles[i]->getInitX();
	}
	// When Gnuplot plots ellipses, it needs the regular axis, not the semi-axis of the cells
	double L = particles[0]->getLength();
	double W = particles[0]->getWidth();
	//printf("length = %f \n",L);
	//printf("width = %f \n",W);
	double A1 = 0;
	A1 = 2.0*L;
	double B1 = 0;
	B1 = 2.0*W;
	
	if(GNUPLOT_PIPE) {
		#ifdef _WIN32
		gnuplotpipe = _popen("pgnuplot -persist", "w");
		std::fprintf(gnuplotpipe,"set term windows\n");
		#elif defined __linux__ || defined __APPLE__&&__MACH__
		gnuplotpipe = popen("gnuplot -persist", "w");
		std::fprintf(gnuplotpipe,"set term x11\n");
		#endif
		std::fprintf(gnuplotpipe,"set size ratio -1\n");
		std::fprintf(gnuplotpipe,"unset key\n");
	}

	//
	// Start simulation time steps
	//
	for(t = 0; t< tMax; t++) // time-steps
	{	
		if(DEBUGGING)
		{
			printf("t=%d\n",t);
			printf("\n");
		}
		noOverlaps=0;
		for(i=0; i<maxSize; i++) // for every cell calculate interactions and speeds first
		{			
			sumX = 0;
			sumY = 0;
			sumPotential = 0;
			sumOverlap = 0;
			for(j=0; j<maxSize; j++) // for all surrounding cells
			{
				potential = 0;
				if(j!=i) // not the same cell
				{
					//potential = Potential(particles[i], particles[j], lamda1, rescale, false);
					overlap = Overlap(particles[i], particles[j]);
					//printf("overlap of particles %i and %i is %f",i,j,overlap);
					if(overlap>0.01*pi*particles[i]->getLength()*particles[i]->getWidth()) //when the overlap area is smaller than 1% of ellipse area it is considered not overlapping
					{	
						//printf("overlap of particles %i and %i is %f\n",i,j,overlap);
						//printf("overlap larger than %f\n",0.1*pi*particles[i]->getLength()*particles[i]->getWidth());
					//	if(overlap>0.05*pi*particles[i]->getLength()*particles[i]->getWidth())
					//	{
							potential = Potential(particles[i], particles[j], lamda1, rescale, false);
					//	}
					//	else
					//	{
					//		potential = Potential(particles[i], particles[j], lamda1, 1., false);
					//	}	
						sumX += eigenX(particles[i],particles[j])*potential;
						sumY += eigenY(particles[i],particles[j])*potential;
						sumPotential += potential;

						sumOverlap++;
					}
					
					//printf("Potential between cell %d and cell %d is %f\n", particles[i]->getID(), particles[j]->getID(), potential);
				}
			}
		//	printf("Summed Potentials = %f\n",sumPotential);
			if (sumOverlap!=0)
			{
				noOverlaps = 1;
				//printf("detected Overlap for Cell i ");
			}
			particles[i]->setSumF(sumPotential);

			//double noiseX = (double)rand()/(double)RAND_MAX - 0.5;
			////double accX = -overdamp*particles[i]->getPrevVelX() + vecNoiseFactor*noiseX + sumX;
		//	double accX = -overdamp*particles[i]->getPrevVelX() + (vecNoiseFactor*noiseX / sqrt(dt)) + sumX;
			double accX = -overdamp*particles[i]->getPrevVelX() + sumX;

			particles[i]->setVelX(particles[i]->getPrevVelX() + accX*dt); // update speed X
			//double noiseY = (double)rand()/(double)RAND_MAX - 0.5;
			////double accY = -overdamp*particles[i]->getPrevVelY() + vecNoiseFactor*noiseY + sumY;
		//	double accY = -overdamp*particles[i]->getPrevVelY() + (vecNoiseFactor*noiseY / sqrt(dt)) + sumY;
			double accY = -overdamp*particles[i]->getPrevVelY() + sumY;
			particles[i]->setVelY(particles[i]->getPrevVelY() + accY*dt); // update speed Y
		} // END OF ACC & VEL LOOP
		if(noOverlaps == 0)
		{
		//	printf("No more overlaps detected. Stopping Simulation and saving pos. and  orient. to ");
		//	cout << outpath;
			//printf("\n");
		//	printf("timestep was: ");
		//	cout << t;
			stringstream sfss;
			sfss << fixed << setprecision(3) << distscale;
			sfss << "_w";
        		sfss << fixed << setprecision(2) << ratio;
			sfss << "_bl";
			sfss << fixed << setprecision(1) << bodyl;
			string sf(sfss.str());
			outfilepathpos=outpath+"//pos_d"+sf+".txt";
			outfilepathorient=outpath+"//headings_d"+sf+".txt";
			output2 = fopen(outfilepathpos.c_str(), "w");
			output1 = fopen(outfilepathorient.c_str(),"w");
			for(i=0;i<maxSize;i++)
				{
			//		std::fprintf(output2, "%i ",i);
					std::fprintf(output2, "%f ", particles[i]->getCurrX()); // save positions on individual file
					std::fprintf(output2, "%f\n", particles[i]->getCurrY());
					std::fprintf(output1, "%f ", cos(particles[i]->getTheta()));
					std::fprintf(output1, "%f\n", sin(particles[i]->getTheta()));
				}
				std::fclose(output2); // close file 
				std::fclose(output1);
			break;
		}
		// Rotate the cells.
		rotateCells(particles,lamda1, dt, 1.1);

		if(GNUPLOT_PIPE)
			std::fprintf(gnuplotpipe,"plot '-' with ellipses\n");

		// update current positions and previous speeds and positions
		for(i=0; i< maxSize; i++)
		{
			particles[i]->setCurrX(particles[i]->getPrevX() + (particles[i]->getVelX())*dt); // update position X
			particles[i]->setCurrY(particles[i]->getPrevY() + (particles[i]->getVelY())*dt); // update position Y
			particles[i]->setPrevX(particles[i]->getCurrX()); // update previous X
			particles[i]->setPrevY(particles[i]->getCurrY()); // update previous Y
			particles[i]->setPrevVelX(particles[i]->getVelX()); // and previous velocity X, Y
			particles[i]->setPrevVelY(particles[i]->getVelY());

			// Write to files
			if(GNUPLOT_PIPE){
				std::fprintf(gnuplotpipe,"%f %f %f %f %f\n",
						particles[i]->getCurrX(),particles[i]->getCurrY(),
						A1, B1, particles[i]->getTheta()*180/M_PI);
			}
		} // END OF POSITIONS LOOP

		if(GNUPLOT_PIPE){
			std::fprintf(gnuplotpipe, "e\n");
			fflush(gnuplotpipe);
		}
		if(t==tMax-1)
		{
			printf("More time steps required to eliminate overlaps");
		}
	} // END OF time steps


	//
	// Plots, outputs
	//
	if(GNUPLOT_PIPE){
		#ifdef _WIN32
			_pclose(gnuplotpipe);
		#elif defined __linux__ || defined __APPLE__&&__MACH__
			pclose(gnuplotpipe);
		#endif
	}

	if(LONGSIM){
		//find the final maximum coordinate value to use for gnuplot plotting range
		double maxCoordsX[maxSize];
		for(i=0;i<maxSize;i++)
		{
			maxCoordsX[i] = particles[i]->getCurrX();
		}
		double maxX = *max_element(maxCoordsX, maxCoordsX + maxSize);

		double maxCoordsY[maxSize];
		for(i=0;i<maxSize;i++)
		{
			maxCoordsY[i] = particles[i]->getCurrY();
		}
		double maxY = *max_element(maxCoordsY, maxCoordsY + maxSize);
		// Generate settings and load files for gnuplot
		//std::cout << "GNU-plotting..." << endl;
		FILE *load = fopen("loadfile.gp", "w");
		std::fprintf(load, "iter=iter+1\n");
		std::fprintf(load, "set output sprintf(\"output_%%d.png\", iter)\n");
		std::fprintf(load, "plot sprintf(\"output_%%d.dat\", iter) using 1:2:3:4:5 with ellipses lw 0.3\n");
		std::fprintf(load, "if(iter < n) reread");
		std::fclose(load);
		FILE *settings = fopen("settings.gp", "w");
		std::fprintf(settings,"set term png\n");
		std::fprintf(settings,"set size ratio -1\n");
		std::fprintf(settings,"set nokey\n");
		std::fprintf(settings,"set xrange [-%i:%i]\n",abs((int)maxX)+50,abs((int)maxX)+50);
		std::fprintf(settings,"set yrange [-%i:%i]\n",abs((int)maxY)+50,abs((int)maxY)+50);
		std::fprintf(settings,"iter = 0\n");
		std::fprintf(settings,"n = 10000\n");
		std::fprintf(settings,"load \"loadfile.gp\"");
		std::fclose(settings);
		//system("gnuplot settings.gp");

		// Make video
		//std::cout << "Generating Video..." << endl;
		//ostringstream command;
		//command << "ffmpeg -f image2 -r 1/0.1 -i output_%01d.png -vcodec mpeg4 " << videoName_str << ".avi -loglevel quiet";
		//system(command.str().c_str());

		// Clean Up! (Compress dat files and delete everything afterwards)
	}
}

/*This uses an ostringstream to build a filename using stream operations,
and returns the built std::string. When you pass it to fopen,
you have to give it a const char * rather than a string,
so use the .c_str() member which exists just for this purpose.*/
string make_output_filename(size_t index)
 {
	 ostringstream ss;
	 ss << "output_" << index << ".dat";
	 return ss.str();
 }

bool almostEqual(const double x, const double y)
{
	if (fabs(x-y) < EPSILON)
		return true;
	else
		return false;
}

