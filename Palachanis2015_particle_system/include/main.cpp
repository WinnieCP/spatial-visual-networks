#include "Cell.h"
#include <ctime>
#include <string>

int main(int argc, char* argv[])
{
    //double elapsed = 0;
    int time, initseed, maxSize;
    double ratio, lamda1, overdamp, dt, distscale, bodyl, ll;
	std::string filepath_pos, filepath_orient, outpath;
	// time is max iterations
	// length is the length of the long semi-axis of the cell
	//Lamda1 is the coefficient of the repulsion area of the cells (their main body)

    if(argc < 13)
	{
        printf("Not enough input parameters!\n");
        printf("Usage:\n");
        printf("'program' + 12 Arguments!\n");
        printf("1) time - maximum iterations, (1 to inf)\n");
        printf("2) Initial random seed - any integer\n");
        printf("3) Number of cells in particle system.\n");
        printf("4) width - the ratio between the minor and major semi-axis of the cell, (.1 to 1 to have any biological meaning)\n");
        printf("5) Lamda1 - coefficient of the repulsion area of the cells (their main body) (0.01 - 0.05) \n");
        printf("6) overdamp - coeffiecient that controls cell inertia (0 - 1).\n");
	printf("7) the path to the txt file containing the position data, column1:x, column2:y\n");
	printf("8) the path to the txt file containing the orientation data, column1:hx, column2:hy\n");
	printf("9 ) the path where the positions and orientations are saved after removal of overlaps\n");
	printf("10) the bodylength of the fish\n");
	printf("11) the distscalefactor, factor by how much the original positions are rescaled to change density\n");
	printf("12 the offset of the ellipse along its major axis from the given positions, l=0 no offset, l+-1 the given positions are at the edge of the ellipse (+nose,-tail)");
	printf("13) dt time step for Euler integration (dt=1 by default).\n");
	
        std::cout << std::endl;
		return 0;
	}
	else
	{
		time = atoi(argv[1]);
        initseed = atoi(argv[2]);
        maxSize = atoi(argv[3]);
        ratio = atof(argv[4]);
        lamda1 = atof(argv[5]);
        overdamp = atof(argv[6]);
        filepath_pos = argv[7];
        filepath_orient = argv[8];
	outpath = argv[9];
        bodyl = atof(argv[10]);
        distscale = atof(argv[11]);
	ll= atof(argv[12]);
	if (argc > 13) dt = atof(argv[13]);
	else dt = 1.0;
  }

    move_cells(time, initseed,  maxSize, ratio, lamda1, overdamp, dt, bodyl, distscale, filepath_pos, filepath_orient, outpath, ll);
    
	return 0;
}

