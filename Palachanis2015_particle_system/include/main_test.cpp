#include "Cell.h"
#include <ctime>
#include <string>




int main(int argc, char* argv[])
{
        double xrep[4], yrep[4];
        double PHI_1, A1, B1, H1, K1, PHI_2, A2, B2, H2, K2, overlap;
        std::string filepath_pos, filepath_orient, outpath;
        // time is max iterations
        // length is the length of the long semi-axis of the cell
        //Lamda1 is the coefficient of the repulsion area of the cells (their main body)
        int rtn_rep,  nroots_rep;
        if(argc < 10)
        {
                printf("Not enough input parameters!\n");
                printf("Usage:\n");
                printf("'program' + 10 Arguments!\n");
                printf("1) orientation of ellipse 1 in radians\n");
                printf("2) long semi axis of ellipse1 \n");
                printf("3) shjor\n");
                printf("4) width - the ratio between the minor and major semi-axis of the cell, (.1 to 1 to have any biological meaning)\n");
                printf("5) Lamda1 - coefficient of the repulsion area of the cells (their main body) (0.01 - 0.05) \n");
                printf("6) overdamp - coeffiecient that controls cell inertia (0 - 1).\n");
                        printf("7) the path to the txt file containing the position data, column1:x, column2:y\n");
                        printf("8) the path to the txt file containing the orientation data, column1:hx, column2:hy\n");
                        printf("9) the bodylength of the fish\n");
                        printf("10) the distscalefactor, factor by how much the original positions are rescaled to change density\n");
                        printf("11) dt time step for Euler integration (dt=1 by default).\n");

                std::cout << std::endl;
                return 0;
        }
        else
        {
                PHI_1 = atof(argv[1]);
                A1 = atof(argv[2]);
                B1 = atof(argv[3]);
                H1 = atof(argv[4]);
                K1 = atof(argv[5]);
                PHI_2 = atof(argv[6]);
                A2 = atof(argv[7]);
                B2 = atof(argv[8]);
                H2 = atof(argv[9]);
                K2 = atof(argv[10]);
        }

        overlap=ellipse_ellipse_overlap(PHI_1, A1, B1, H1, K1, PHI_2, A2, B2, H2, K2, xrep, yrep, &nroots_rep, &rtn_rep);
        if(overlap==0 or overlap==-1)
        {	
		std::cout.precision(17);
                printf("No Overlap? Detected: ");
                std::cout<<overlap;
                printf("----------------");
                printf("\nEllipse 1:\n");
                printf("Heading: ");
                std::cout<<PHI_1;
                printf("\nx-Semi Axis (half of length):");
                std::cout<<A1;
                printf("\ny-Semi Axis (half of width): ");
                std::cout<<B1;
                printf("\nx Position: ");
                std::cout<<H1;
                printf("\ny Position: ");
                std::cout<<K1;
                printf("\n...............\n");
                printf("\nEllipse 2:\n");
                printf("Heading: ");
                std::cout<<PHI_2;
                printf("\nx-Semi Axis (half of length):");
                std::cout<<A2;
                printf("\ny-Semi Axis (half of width): ");
                std::cout<<B2;
                printf("\nx Position: ");
                std::cout<<H2;
                printf("\ny Position: ");
                std::cout<<K2;
                printf("\n----------------");
        }
        else
        {
                printf("\n----------------\n");
                printf("Overlap area: ");
                std::cout<<overlap;
                printf("\n----------------\n");
        }
        return 0;
}
