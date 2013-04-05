#include <mex.h>
#include <math.h>
#include <algorithm>
#include <functional>

/*
solves argmax_{h_1,h_2} { Wx1.h_1 + Wx2.h2 + loss(hamm_dis(h1,h2)) }

Input: 
   Wx1, Wx2, Loss

Output:
   H1, H2, Ndiff= Hamming_Dist(H1,H2)
*/

/* Input Arguments */
#define	Wx1   	prhs[0]
#define Wx2     prhs[1]
#define Loss    prhs[2]

/* Output Arguments */
#define	H1	plhs[0]
#define	H2	plhs[1]
#define Ndiff   plhs[2]


using namespace std;

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{
    double *pWx1, *pWx2, *pLoss, *pH1, *pH2, *pNdiff;
    
    /* Check for proper number of input arguments */
    if (nrhs != 3) { 
	mexErrMsgTxt("Three input arguments required."); 
    }

    /* Check for proper number of output arguments */
    if (nlhs != 3) { 
	mexErrMsgTxt("Three output arguments required."); 
    }
        
    /* Get dimensions of inputs */
    int n = (int) mxGetM(Wx1);
    int m = (int) mxGetN(Wx1);
    if (n != (int) mxGetM(Wx2) || m != (int) mxGetN(Wx2)) {
	mexErrMsgTxt("First and Second arguments should be equal size matrices");
	return;
    }
    if (n+1 != (int) mxGetM(Loss) || m != (int) mxGetN(Loss)) {
	mexErrMsgTxt("Third argument does not have appropriate size");
	return;
    }
    pWx1 = (double*) mxGetPr(Wx1);
    pWx2 = (double*) mxGetPr(Wx2);
    pLoss = (double*) mxGetPr(Loss);

    H1 = mxCreateNumericMatrix(n, m, mxDOUBLE_CLASS, mxREAL);
    H2 = mxCreateNumericMatrix(n, m, mxDOUBLE_CLASS, mxREAL);
    Ndiff = mxCreateNumericMatrix(1, m, mxDOUBLE_CLASS, mxREAL);
    pH1 = (double*) mxGetPr(H1);
    pH2 = (double*) mxGetPr(H2);
    pNdiff = (double*) mxGetPr(Ndiff);
    
    double *same = NULL;
    double *diff = NULL;
    int *sameInd = NULL;
    int *diffInd = NULL;
    pair<double,int> *diff_vs_same = NULL;

    int j = 0;

#pragma omp parallel shared(j) private(same, diff, sameInd, diffInd, diff_vs_same, pWx1, pWx2, pLoss, pH1, pH2, pNdiff)
    {
	same = new double [n];
	diff = new double [n];
	sameInd = new int [n];
	diffInd = new int [n];
	diff_vs_same = new pair<double,int> [n];

#pragma omp for
	for (j=0; j<m; j++) {
	    pWx1 = mxGetPr(Wx1) + j*n;
	    pWx2 = mxGetPr(Wx2) + j*n;
	    pLoss = mxGetPr(Loss) + j*(n+1);
	    pH1 = mxGetPr(H1) + j*n;
	    pH2 = mxGetPr(H2) + j*n;
	    pNdiff = mxGetPr(Ndiff);

	    for (int i=0; i<n; i++) {
		double samePos = pWx1[i] + pWx2[i];
		if (samePos > -samePos) {
		    same[i] = samePos;
		    sameInd[i] = 0;
		} else {
		    same[i] = -samePos;
		    sameInd[i] = 1;
		}

		double diffPosNeg = pWx1[i] - pWx2[i];	    
		if (diffPosNeg > -diffPosNeg) {
		    diff[i] = diffPosNeg;
		    diffInd[i] = 0;
		} else {
		    diff[i] = -diffPosNeg;
		    diffInd[i] = 1;
		}

		diff_vs_same[i].first = diff[i] - same[i];
		diff_vs_same[i].second = i;
	    }

	    sort(diff_vs_same, diff_vs_same + n, greater<pair<double, int> >() );

	    double uptonow = 0;
	    double best_score = pLoss[0];
	    double best_ndiff = 0;

	    for (int i=0; i<=n; i++) {
		double cur = pLoss[i] + uptonow;
		if (cur > best_score) {
		    best_score = cur;
		    best_ndiff = i;
		}
		if (i < n)
		    uptonow += diff_vs_same[i].first;
	    }
	    pNdiff[j] = best_ndiff;

	    for (int i=0; i<best_ndiff; i++) {
		int j = diff_vs_same[i].second;
		pH1[j] = diffInd[j] ? -1 : 1;
		pH2[j] = diffInd[j] ? 1  : -1;
	    }
	    for (int i=best_ndiff; i<n; i++) {
		int j = diff_vs_same[i].second;
		pH1[j] = sameInd[j] ? -1 : 1;
		pH2[j] = sameInd[j] ? -1 : 1;
	    }
	}

	delete[] same;
	delete[] diff;
	delete[] sameInd;
	delete[] diffInd;
	delete[] diff_vs_same;
    }
  
    return;
}
