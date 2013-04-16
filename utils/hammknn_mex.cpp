#include <mex.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <time.h>

#include "types.h"
#include "linscan.h"

// INPUTS --------------------

#define mxnlabels       prhs[0] // number of class labels
#define mxcodes		prhs[1] // db binary codes
#define mxlabels        prhs[2] // class labels for db codes
#define mxqueries 	prhs[3] // query binary codes
#define mxlabels2       prhs[4] // class labels for query codes
#define mxN		prhs[5]	// number of data points to use
#define mxB		prhs[6]	// number of bits per code
#define mxK		prhs[7]	// k in hamming kNN classifier
// pointers labels == labels2 being identical means that this is a 1-leave-out cross validation for the training set.


// OUTPUTS --------------------

#define mxcorrect	plhs[0]
#define mxcorrect2	plhs[1]

void myAssert(int a, const char *b) {
    if (!a)
	mexErrMsgTxt(b);
}

void classify(unsigned int *correct, int maxk, int K, UINT32 *counter, int B, UINT32 *res, unsigned int *labels, unsigned int *labels_test, int NQ, int nlabels) {
    int *label_count = new int[nlabels];
    for (int i=0; i<NQ; i++) {
	int dis = -1;
	memset(label_count, 0, 10*sizeof(*label_count));
	int cum_nn = 0;
	int new_cum_nn = 0;
	int lastk = 0;
	int itsown = 0;
	
	for (int hdis=0; hdis<=B; hdis++) {
	    new_cum_nn = cum_nn + counter[(B+1)*i + hdis];
	    if (new_cum_nn > K)
		new_cum_nn = K;
	    
	    for (int j=cum_nn; j<new_cum_nn; j++) {
		int neighbor = res[i*K+j]; // make sure res is 0-based
 		if (labels == labels_test && neighbor == i) {
		    itsown = 1;
		    lastk++; // we will subtract "itsown" from "k" later, so lask++
		} else
		    label_count[labels[neighbor]]++;
	    }

	    bool tie = 0;
	    int pred = 0;
	    int count_pred = 0;
	    for (int j=0; j<nlabels; j++)
		if (count_pred < label_count[j]) {
		    count_pred = label_count[j];
		    pred = j;
		    tie = 0;
		} else if (count_pred == label_count[j]) { // ties
		    tie = 1;
		}
	    if (!tie || new_cum_nn == K) { // if it is not a tie, or the max number of neighbors is reached, we will update number of correct predictions for values of k from lastk up to min(maxk, new_cum_nn).
		for (int k=lastk; k<std::min(maxk+itsown, new_cum_nn); k++) {
		    if (labels_test[i] == pred)
			correct[k-itsown]++;
		}
		lastk = std::min(maxk+itsown, new_cum_nn);
	    }

	    if (lastk == maxk+itsown)
		break;
	    cum_nn = new_cum_nn;
	}
    }
    delete[] label_count;
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )    
{
    if (nrhs != 8)
	mexErrMsgTxt("Wrong number of inputs\n");
    if (nlhs != 2)
	mexErrMsgTxt("Wrong number of outputs\n");
    
    int N = (int)(UINT32) *(mxGetPr(mxN));
    int B = (int) *(mxGetPr(mxB));
    int maxk = (int) *(mxGetPr(mxK));
    int nlabels = (int) *(mxGetPr(mxnlabels));
    
    UINT8 *codes_db = (UINT8*) mxGetPr(mxcodes);
    unsigned int *label_db = (unsigned int *) (UINT32*) mxGetPr(mxlabels);
    UINT8 *codes_query = (UINT8*) mxGetPr(mxqueries);
    unsigned int *label_query = (unsigned int *) (UINT32*) mxGetPr(mxlabels2);
	
    int NQ = mxGetN(mxqueries);
    int dim1codes = mxGetM(mxcodes);
    int dim1queries = mxGetM(mxqueries);

    myAssert(mxIsUint8(mxqueries), "queries is not uint8");
    myAssert(mxIsUint8(mxcodes), "codes is not uit8");
    myAssert(mxIsUint32(mxlabels), "labels is not uint32");
    myAssert(mxGetN(mxcodes) >= N, "number of codes < N");
    myAssert(dim1codes >= B/8, "dim1codes < B/8");
    myAssert(dim1queries >= B/8, "dim1queries < B/8");	// not sure about this
    myAssert(B % 8 == 0, "mod(B,8) != 0");

    myAssert(dim1codes == B/8, "dim1codes != B/8");
    myAssert(dim1queries == B/8, "dim1queries != B/8");	// not sure about this

    mxcorrect = mxCreateNumericMatrix(maxk, 1, mxUINT32_CLASS, mxREAL);
    mxcorrect2 = mxCreateNumericMatrix(maxk, 1, mxUINT32_CLASS, mxREAL);
    unsigned int *correct  = (unsigned int*) (UINT32 *) mxGetPr(mxcorrect);
    unsigned int *correct2 = (unsigned int*) (UINT32 *) mxGetPr(mxcorrect2);

    int K = maxk*5 + 100;
    if (K > N)
	K = N;
    UINT32 *counter = new UINT32[(B+1)*NQ];
    UINT32 *res = new UINT32[K*NQ];

    linscan_query(counter, res, codes_db, codes_query, N, NQ, B, K, B/8, B/8);
    classify(correct, maxk, K, counter, B, res, label_db, label_query, NQ, nlabels);
    
    delete[] counter;
    delete[] res;
    counter = new UINT32[(B+1)*N];
    res = new UINT32[K*N];

    linscan_query(counter, res, codes_db, codes_db, N, N, B, K, B/8, B/8);
    classify(correct2, maxk, K, counter, B, res, label_db, label_db, N, nlabels);
    
    delete[] counter;
    delete[] res;
}
