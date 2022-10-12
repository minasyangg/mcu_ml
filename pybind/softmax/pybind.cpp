#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


float* get_batch(const float* X, int start_pos, int n, int bacth_size)
{
	float* mini_batch = new float[bacth_size*n];
	for (int i=0; i<bacth_size; i++)
		for (int j=0; j<n; j++)
			mini_batch[i*n + j] = X[n*(i+start_pos) + j];

	return mini_batch;
}

float* dot_matr(float* A, float* B, int m, int n, int k){
	float* matr_out = new float[m*k];
	for (int i=0; i<m; i++)
		for(int j=0; j<k; j++){
			float matr_elem = 0;
				for(int z=0; z<n; z++){
					matr_elem += A[i*n+z]*B[z*k+j];
					// printf("%f", matr_elem);
				}
				matr_out[i*k+j] = matr_elem;
		}
	return matr_out;
}

float* rez_dot(const float* A, const float* B, int n, int batch, int k){
	float* matr_out = new float[n*k];
	for (int i=0; i<n; i++)
		for(int j=0; j<k; j++){
			float matr_elem = 0;
				for(int z=0; z<batch; z++){
					matr_elem += A[i*batch+z]*B[k*z+j];
				}
				matr_out[i*k+j] = matr_elem;
				
	}
	return matr_out;
}


float* transpose(float* batch_x, int batch, int n)
{
	float* x_b_T = new float[batch*n];
	for (int i=0; i<batch; i++)
		for(int j=0; j<n; j++){
			x_b_T[j*batch+i] = batch_x[i*n+j];
	}

	return x_b_T;
}

float* softmax(float* dot_x_b_theta, int batch, int n, int k){
	float* raw_sum = new float[batch];
	float* Z = new float[batch*k];
	for (int i=0; i<batch; i++){
		raw_sum[i] = 0;
		for (int j=0; j<k; j++){
			raw_sum[i] += exp(dot_x_b_theta[i*k+j]);
			dot_x_b_theta[i*k+j] = exp(dot_x_b_theta[i*k+j]);
		}	
	}

	float* dot_x_b_theta_T = transpose(dot_x_b_theta, batch, k);



	for (int i=0; i<k; i++)
		for (int j=0; j<batch; j++){
			dot_x_b_theta_T[i*batch+j] /= raw_sum[j];
		}
		
	Z = transpose(dot_x_b_theta_T, batch, k);
	delete[] raw_sum;
	return Z;
	
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
														float *theta, size_t m, size_t n, size_t k,
														float lr, size_t batch)
{
	int num_batches = int (m/ batch);
	// printf("%d", num_batches);
	for (int i=0; i<num_batches; i++)	{
		float* X_b = get_batch(X, i*batch, n, batch);
		char* y_b = new char[batch];
		for (int j=0;j<batch;j++)
			y_b[j] = y[i*batch+j];

		float* X_b_T = new float[batch*n];
		X_b_T = transpose(X_b, batch, n);
		float* dot_x_b_theta = new float[m*k];
		dot_x_b_theta = dot_matr(X_b, theta, batch, n, k);

		float* Z = new float[k*batch];
		Z = softmax(dot_x_b_theta, batch, n, k);
		for (int i=0; i<batch; i++)
			Z[i*k+y_b[i]] -= 1;
		
		float* result_matr = new float[n*k];
		result_matr = rez_dot(X_b_T, Z, n, batch, k);

		for (int i=0; i<n; i++)
			for(int j=0; j<k; j++){
				theta[i*k+j] -= (lr/batch) * result_matr[i*k+j];
			}
		
			float norm = 0;
			for(int i=0; i<n; i++)
				for (int j=0; j<k; j++){
					norm += theta[i*k+j]*theta[i*k+j];
			}

	printf("%f\n", sqrt(norm));

		delete[] X_b;
		delete[] y_b;
		delete[] X_b_T;
		delete[] result_matr;
		delete[] dot_x_b_theta;
		delete[] Z;
	}
}




/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(mymodule, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
