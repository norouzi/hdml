#include <mex.h>
#include <math.h>
#include <algorithm>
#include <functional>

/*
try to solve argmax_h { Wx1.h1 + Wx2.h2 + Wx3.h3 + loss(nb + hamm_dis(h1,h2)-hamm_dis(h1,h3)) }

Input: 
   Wx1, Wx2, Wx3, Loss

Output:
   H1, H2
*/
/* Input Arguments */
#define	Wx1   	prhs[0]		// f(x), n x m, n is number of bits, m is number of points, column major
#define Wx2     prhs[1]		// f(x+)
#define Wx3     prhs[2]		// f(x-)
#define Loss    prhs[3]		// 2n+1 rows, m columns.. why?

/* Output Arguments */
#define	H1	plhs[0]		// g
#define	H2	plhs[1]		// g+
#define	H3	plhs[2]		// g-

#define Diff    plhs[3]		// hamm_dis(h1,h2)-hamm_dis(h1,h3)
#define WxLoss  plhs[4]		// Wx1.h1 + Wx2.h2 + Wx3.h3
// #define Wx

using namespace std;

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{
    double *pWx1, *pWx2, *pWx3, *pLoss, *pH1, *pH2, *pH3, *pDiff, *pWxLoss;
    
    /* Check for proper number of input arguments */    
    if (nrhs != 4) { 
	mexErrMsgTxt("Four input arguments required."); 
    }

    /* Check for proper number of output arguments */    
    if (nlhs != 5) { 
	mexErrMsgTxt("Five output arguments required."); 
    }
        
    /* Get dimensions of inputs */
    int n = (int) mxGetM(Wx1);
    int m = (int) mxGetN(Wx1);
    if (n != (int) mxGetM(Wx2) || m != (int) mxGetN(Wx2)) {
	mexErrMsgTxt("First and second arguments should be equal size matrices");
	return;
    }
    if (n != (int) mxGetM(Wx3) || m != (int) mxGetN(Wx3)) {
	mexErrMsgTxt("First and third arguments should be equal size matrices");
	return;
    }
    if (2*n+1 != (int) mxGetM(Loss) || m != (int) mxGetN(Loss)) {
	mexErrMsgTxt("Fourth argument does not have appropriate size");
	return;
    }
    pWx1 = mxGetPr(Wx1);
    pWx2 = mxGetPr(Wx2);
    pWx3 = mxGetPr(Wx3);
    pLoss = mxGetPr(Loss);

    H1 = mxCreateNumericMatrix(n, m, mxDOUBLE_CLASS, mxREAL);
    H2 = mxCreateNumericMatrix(n, m, mxDOUBLE_CLASS, mxREAL);
    H3 = mxCreateNumericMatrix(n, m, mxDOUBLE_CLASS, mxREAL);
    pH1 = (double*) mxGetPr(H1);
    pH2 = (double*) mxGetPr(H2);
    pH3 = (double*) mxGetPr(H3);

    Diff = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
    WxLoss = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
    pDiff = mxGetPr(Diff);
    pWxLoss = mxGetPr(WxLoss);
    
    double **cost = NULL;
    int **costInd = NULL;
    double **table = NULL;
    int **tableInd = NULL;
    int j = 0;

#pragma omp parallel shared(j) private(cost, costInd, table, tableInd, pWx1, pWx2, pWx3, pLoss, pH1, pH2, pH3, pDiff, pWxLoss)
    {
	cost = new double* [n];
	for (int i=0; i<n; i++) {
	    cost[i] = new double[3];
	    cost[i]++;
	}

	costInd = new int* [n];
	for (int i=0; i<n; i++) {
	    costInd[i] = new int[3];
	    costInd[i]++;
	}

	table = new double* [n+1];
	for (int i=0; i<=n; i++) {
	    table[i] = new double[2*n+1];
	    table[i] += n;
	}

	tableInd = new int* [n+1];
	for (int i=0; i<=n; i++) {
	    tableInd[i] = new int[2*n+1];
	    tableInd[i] += n;
	}

	#pragma omp for
	for (j=0; j<m; j++) {
	    pWx1 = mxGetPr(Wx1) + j*n;
	    pWx2 = mxGetPr(Wx2) + j*n;
	    pWx3 = mxGetPr(Wx3) + j*n;
	    pLoss = mxGetPr(Loss) + j*(2*n+1);
	    
	    pH1 = mxGetPr(H1) + j*n;
	    pH2 = mxGetPr(H2) + j*n;
	    pH3 = mxGetPr(H3) + j*n;
	    pDiff = mxGetPr(Diff) + j;
	    pWxLoss = mxGetPr(WxLoss) + j;
	    
	    for (int i=0; i<n; i++) {
		cost[i][0] = pWx1[i] + pWx2[i] + pWx3[i];
		costInd[i][0] = 0;
		if (cost[i][0] < -pWx1[i] -pWx2[i] -pWx3[i]) {
		    cost[i][0] = -pWx1[i] -pWx2[i] -pWx3[i];
		    costInd[i][0] = 1;
		}
		if (cost[i][0] < +pWx1[i] -pWx2[i] -pWx3[i]) {
		    cost[i][0] = +pWx1[i] -pWx2[i] -pWx3[i];
		    costInd[i][0] = 2;
		}
		if (cost[i][0] < -pWx1[i] +pWx2[i] +pWx3[i]) {
		    cost[i][0] = -pWx1[i] +pWx2[i] +pWx3[i];
		    costInd[i][0] = 3;
		}

		cost[i][+1] = +pWx1[i] -pWx2[i] +pWx3[i];
		costInd[i][+1] = 0;
		if (cost[i][+1] < -cost[i][+1]) {
		    cost[i][+1] = -cost[i][+1];
		    costInd[i][+1] = 1;	
		}

		cost[i][-1] = +pWx1[i] +pWx2[i] -pWx3[i];
		costInd[i][-1] = 0;
		if (cost[i][-1] < -cost[i][-1]) {
		    cost[i][-1] = -cost[i][-1];
		    costInd[i][-1] = 1;
		}

		// printf("%.3f %.3f %.3f\n", cost[i][-1], cost[i][0], cost[i][+1]);
		// printf("%d %d %d\n", costInd[i][-1], costInd[i][0], costInd[i][+1]);
	    }

	    table[0][0] = 0;
	    tableInd[0][0] = -2;

	    for (int i=1; i<=n; i++) {
		table[i][-i] = -1e100;
		table[i][i] = -1e100;
		for (int k=-(i-1); k<=+(i-1); k++) {
		    table[i][k] = table[i-1][k] + cost[i-1][0];
		    tableInd[i][k] = 0;
		}
		for (int k=-i; k<=i-2; k++)
		    if (table[i][k] < table[i-1][k+1] + cost[i-1][-1]) {
			table[i][k] = table[i-1][k+1] + cost[i-1][-1];
			tableInd[i][k] = -1;
		    }
		for (int k=-(i-2); k<=i; k++)
		    if (table[i][k] < table[i-1][k-1] + cost[i-1][+1]) {
			table[i][k] = table[i-1][k-1] + cost[i-1][+1];
			tableInd[i][k] = +1;
		    }
	    }
	
	    // printf("%.2f\n", table[n][0]);
	    int cur = 0;
	    double cur_best = -1e100;
	    pLoss += n;
	    for (int mag_k=0; mag_k<=n; mag_k++)
		for (int sign_k=-1; sign_k<=1; sign_k+=2) {
		    int k = mag_k*sign_k;
		    if (cur_best < pLoss[k] + table[n][k]) {
			cur_best = pLoss[k] + table[n][k];
			cur = k;
		    }
		}
	    *pDiff = cur;
	    *pWxLoss = cur_best;

	    int cur2, cur3;
	    for (int i=n; i>0; i--) {
		cur2 = tableInd[i][cur];
		cur3 = costInd[i-1][cur2];
		if (cur2 == 0) {
		    if (cur3 == 0) {
			pH1[i-1] = 1; pH2[i-1] = 1; pH3[i-1] = 1;
		    } else if (cur3 == 1) {
			pH1[i-1] = -1; pH2[i-1] = -1; pH3[i-1] = -1;
		    } else if (cur3 == 2) {
			pH1[i-1] = +1; pH2[i-1] = -1; pH3[i-1] = -1;
		    } else if (cur3 == 3) {
			pH1[i-1] = -1; pH2[i-1] = +1; pH3[i-1] = +1;
		    }
		} else if (cur2 == +1) {
		    if (cur3 == 0) {
			pH1[i-1] = +1; pH2[i-1] = -1; pH3[i-1] = +1;
		    } else if (cur3 == 1) {
			pH1[i-1] = -1; pH2[i-1] = +1; pH3[i-1] = -1;
		    }
		} else if (cur2 == -1) {
		    if (cur3 == 0) {
			pH1[i-1] = +1; pH2[i-1] = +1; pH3[i-1] = -1;
		    } else if (cur3 == 1) {
			pH1[i-1] = -1; pH2[i-1] = -1; pH3[i-1] = +1;
		    }
		}
		cur -= cur2;
	    }
	
	    // pWx1 += n;
	    // pWx2 += n;
	    // pWx3 += n;
	    // pLoss += (-n) + 2*n+1;
	    // pH1 += n;
	    // pH2 += n;
	    // pH3 += n;
	}

	for (int i=0; i<n; i++) {
	    cost[i]--;
	    delete[] cost[i];
	}
	delete[] cost;

	for (int i=0; i<n; i++) {
	    costInd[i]--;
	    delete[] costInd[i];
	}
	delete[] costInd;
  
	for (int i=0; i<=n; i++) {
	    table[i] -= n;
	    delete[] table[i];
	}
	delete[] table;

	for (int i=0; i<=n; i++) {
	    tableInd[i] -= n;
	    delete[] tableInd[i];
	}
	delete[] tableInd;
    }

    return;
}
