#define CLASSIFICATION 1
#define REGRESSION 2
#define CLIP_VALUE 100

#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
using namespace Rcpp;
using namespace std;

int omp_thread_count() {
  int n = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:n)
#endif
  n += 1;
  return n;
}

inline float clip(float x) {
  float sign = x < 0.0 ? -1.0:1.0;
  return(fabs(x) < CLIP_VALUE ? x : sign * CLIP_VALUE);
}

class FMParam {
public:
  FMParam();
  FMParam(float learning_rate,
          int rank,
          float lambda_w, float lambda_v,
          std::string task_name):
    learning_rate(learning_rate),
    rank(rank),
    w0(0),
    lambda_w(lambda_w),
    lambda_v(lambda_v) {
    if ( task_name == "classification")
      this->task = CLASSIFICATION;
    else if( task_name == "regression")
      this->task = REGRESSION;
    else throw(Rcpp::exception("can't match task code - not in (1=CLASSIFICATION, 2=REGRESSION)"));
  }
  int task = 0;

  float learning_rate;

  int n_features;
  int rank;
  float w0;

  float lambda_w;
  float lambda_v;
  float lambda_w0;


  vector< float > w;
  vector< vector< float > > v;

  //squared gradients for adaptive learning rate
  vector< vector< float > > grad_v2;
  vector< float > grad_w2;

  float link_function(float x) {
    if(this->task == CLASSIFICATION)
      return(1.0 / ( 1.0 + exp(-x)));
    if(this->task == REGRESSION)
      return(x);
    return(x);
    throw(Rcpp::exception("no link function"));
  }
  float loss(float pred, float actual) {

    if(this->task == CLASSIFICATION)
      return(-log( this->link_function(pred * actual) ));

    if(this->task == REGRESSION)
      return((pred - actual) * (pred - actual));

    return(-log( this->link_function(pred * actual) ));
    throw(Rcpp::exception("no loss function"));
  }
  List dump();
  void init_weights(const NumericVector &w_R, const NumericMatrix &v_R);
};

List FMParam::dump() {
  NumericVector w_dump(n_features);
  NumericMatrix v_dump(n_features, rank);
  for(int i = 0; i < n_features; i++) {
    w_dump[i] = w[i];
    for(int j = 0; j < rank; j++)
      v_dump(i, j) = v[i][j];
  }
  return(List::create(_["w"] = w_dump, _["v"] = v_dump));
}

void FMParam::init_weights(const NumericVector &w_R, const NumericMatrix &v_R) {
  // number of features equal to number of input weights
  this->n_features = w_R.size();
  w.resize(this->n_features);
  grad_w2.resize(this->n_features);

  for(int i = 0; i < n_features; i++) {
    // init single feature weights
    w[i] = w_R[i];
    // gradient square history for single feature weights
    grad_w2[i] = 1;
  }
  v.resize(n_features);
  grad_v2.resize(n_features);
  for(int k = 0; k < n_features; k++) {
    // interactions vectors
    v[k].resize(rank);
    // gradient history for interactions weights
    grad_v2[k].resize(rank);

    for(int i = 0; i < rank; i ++) {
      // init with interactions weights
      v[k][i] = v_R(k, i);
      // init gradient square history
      grad_v2[k][i] = 1;
    }
  }
}

class FMModel {
public:
  FMModel(FMParam *params): params(params) {};
  FMParam *params;

  float fm_predict_internal(int *nnz_index, const vector<float> &nnz_value, int offset_start, int offset_end) {
    //float res = 0.0;
    float res = this->params->w0;
    // add linear terms
    for(int j = offset_start; j < offset_end; j++) {
      int feature_index = nnz_index[j];
      res += this->params->w[feature_index] * nnz_value[j];
    }
    float res_pair_interactions = 0.0;
    // add interactions
    for(int f = 0; f < this->params->rank; f++) {
      float s1 = 0.0;
      float s2 = 0.0;
      float prod;
      for(int j = offset_start; j < offset_end; j++) {
        int feature_index = nnz_index[j];
        prod = this->params->v[feature_index][f] * nnz_value[j];
        s1  += prod;
        s2  += prod * prod;
      }
      res_pair_interactions += s1 * s1 - s2;
    }
    return(res + 0.5 * res_pair_interactions);
  }

  NumericVector fit_predict(const S4 &m, const NumericVector &y_R, const NumericVector &w_R, int nthread = 0, int do_update = 1) {
    int nth = omp_thread_count();
    const double *y = y_R.begin();
    const double *w = w_R.begin();
    // override if user manually specified number of threads
    if(nthread > 0)
      nth = nthread;

    IntegerVector dims = m.slot("Dim");
    // numer of sample in current mini-batch
    int N = dims[0];
    // allocate space for result
    NumericVector y_hat_R(N);
    // get pointers to not touch R API
    double *y_hat = y_hat_R.begin();

    // just to extract vectors from S4
    // and get pointers to data - we can't touch R API in threads, so will use raw pointers
    IntegerVector PP = m.slot("p");
    int *P = PP.begin();
    IntegerVector JJ = m.slot("j");
    int *J = JJ.begin();
    NumericVector XX = m.slot("x");
    // copy to float vector to improve SIMD performance
    vector<float> X(XX.size());
    for(int i = 0; i < XX.size(); i++)
      X[i] = XX[i];

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nth)
    #endif
    for(int i = 0; i < N; i++) {
      size_t p1 = P[i];
      size_t p2 = P[i + 1];
      float y_hat_raw = this->fm_predict_internal(J, X, p1, p2);
      // prediction
      y_hat[i] = this->params->link_function(y_hat_raw);
      // fitting
      if(do_update) {
        //------------------------------------------------------------------
        // first part of d_L/d_theta -  intependent of parameters theta
        float dL;
        if(this->params->task == CLASSIFICATION)
          dL = (this->params->link_function(y_hat_raw * y[i]) - 1) * y[i];
        else if(this->params->task == REGRESSION )
          dL = 2 * (y_hat_raw - y[i]);
        else
         throw(Rcpp::exception("task not defined in FMModel::fit_predict()"));
        // mult by error-weight of the sample
        dL *= w[i];
        //------------------------------------------------------------------
        // update w0
        this->params->w0 -= this->params->learning_rate * dL;

        vector<float> grad_v_k(this->params->rank);
        for( int p = p1; p < p2; p++) {

          int   feature_index  = J[p];
          float feature_value = X[p];

          float grad_w = clip(feature_value * dL + 2 * this->params->lambda_w);

          this->params->w[feature_index] -= this->params->learning_rate * grad_w / sqrt(this->params->grad_w2[feature_index]);
          // update sum gradient squre
          this->params->grad_w2[feature_index] += grad_w * grad_w;

          // pairwise interactions
          //------------------------------------------------------------------------
          // SIMD vectorized inner products
          // this chunk extracted from inside of next main loop
          // iteration through factors and non-zero elements reordered
          //------------------------------------------------------------------------
          int len = p2 - p1;
          for(int f = 0; f < this->params->rank; f++)
            grad_v_k[f] = -this->params->v[feature_index][f] * feature_value;

          for(int k = 0; k < len; k++) {
            float val = X[p1 + k];
            int index = J[p1 + k];
            float *v_ptr = &this->params->v[index][0];

            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for(int f = 0; f < this->params->rank; f++)
              grad_v_k[f] += v_ptr[f] * val;
          }
          //------------------------------------------------------------------------
          #ifdef _OPENMP
          #pragma omp simd
          #endif
          for(int f = 0; f < this->params->rank; f++) {
            // THIS CHUNK was moved out of the loop in order to vectorize it with SIMD
            //------------------------------------------------------------------------
            // float grad_v = 0;
            // for(int j = p1; j < p2; j++) {
            //   int feature_index_2 = J[j];
            //   float feature_value_2 = X[j];
            //     grad_v += this->params->v[feature_index_2][f] * feature_value_2;
            // }
            // grad_v = feature_value * (grad_v - this->params->v[feature_index][f] * feature_value);
            // grad_v += 2 * this->params->v[feature_index][f] * this->params->lambda_v;
            // grad_v = clip(grad_v * dL, 100);
            //------------------------------------------------------------------------
            float grad_v = dL * (feature_value * grad_v_k[f] + 2 * this->params->v[feature_index][f] * this->params->lambda_v);
            // clip for numerical stability
            grad_v = clip(grad_v);
            // update params
            this->params->v[feature_index][f] -= this->params->learning_rate * grad_v / sqrt(this->params->grad_v2[feature_index][f]);
            // update sum gradient squre
            this->params->grad_v2[feature_index][f] += grad_v * grad_v;
          }
        }
      }
    }
    return(y_hat_R);
  }
};

// [[Rcpp::export]]
SEXP fm_create_param(float learning_rate,
                     int rank, float lambda_w, float lambda_v, const String task) {
  FMParam * param = new FMParam(learning_rate, rank, lambda_w, lambda_v, task);
  XPtr< FMParam> ptr(param, true);
  return ptr;
}

// [[Rcpp::export]]
void fm_init_weights(SEXP ptr, const NumericVector &w_R, const NumericMatrix &v_R) {
  Rcpp::XPtr<FMParam> params(ptr);
  params->init_weights(w_R, v_R);
}

// [[Rcpp::export]]
NumericVector fm_partial_fit(SEXP ptr, const S4 &X, const NumericVector &y, int nthread = 1, int do_update = 1) {
  Rcpp::XPtr<FMParam> params(ptr);
  FMModel model(params);
  return(model.fit_predict(X, y, nthread, do_update));
}

// [[Rcpp::export]]
List fm_dump(SEXP ptr) {
  Rcpp::XPtr<FMParam> params(ptr);
  FMModel model(params);
  return(model.params->dump());
}
