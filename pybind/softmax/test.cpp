#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <assert.h>

int main (void) {
	int m = 50;
	int n = 5;
	int k = 3;
	float lr = 0.2;
	int batch = 10;
	float X[m][n];
	int y[m];
	float theta[m][k];
	
	srand(0);
	for (int i=0;i<m;i++)
		y[i] = (int)rand()/((int)RAND_MAX/3);

	for (int i=0;i<n;i++)
		for (int j=0; j<10; j++)
			theta[i][j] = (float)rand()/((float)RAND_MAX/2);



	
	int count = 0;
	for (int i=0; i<m;i+=batch)
		{
			float x_b[batch][n];
			int y_b[batch];
			float i_b[batch][k];

			for (int j=0; j<batch;j++)
			{
				y_b[j] = y[j+count*batch];
				for(int z=0; z<n;z++)
					{
						x_b[j][z] = X[j+count*batch][z];
					}
			}
			for (int j=0;j<batch;j++)
				for(int a=0;a<k;a++)
					{
						if (a == y_b[j])
							{
								i_b[j][a] = 1;
							}
						else
							{
								i_b[j][a] = 0;
							}
					}
	


		count++;	
		printf("\n");		
		}


}








