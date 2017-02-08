#' @useDynLib FM
#' @importFrom Rcpp evalCpp
NULL
#' @export
FM = R6::R6Class(
  classname = "estimator",
  public = list(
    #-----------------------------------------------------------------
    initialize = function(learning_rate = 0.2, rank = 4,
                          lambda_w = 0, lambda_v = 0,
                          family = c("binomial")) {
      stopifnot(lambda_w >= 0 && lambda_v >= 0 && learning_rate > 0 && rank >= 1)
      family = match.arg(family);
      private$init_model_param(learning_rate = learning_rate, rank = rank,
                               lambda_w = lambda_w, lambda_v = lambda_v,
                               family = family)
    },
    partial_fit = function(X, y, nthread = 0, ...) {
      if(!inherits(class(X), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix (class ", class(X), ") to ", private$internal_matrix_format)
        X = as(X, private$internal_matrix_format)
      }
      X_ncol = ncol(X)
      # init model during first first fit
      if(!private$is_initialized) {
        private$init_model_state(n_features = X_ncol, NULL, NULL)
      }
      # on consequent updates check that we are wotking with input matrix with same numner of features
      stopifnot(X_ncol == private$n_features)
      # check number of samples = number of outcomes
      stopifnot(nrow(X) == length(y))

      stopifnot(!anyNA(y))
      # convert to (1, -1) as it required by loss function in FM
      if(private$family == 'binomial')
        y = ifelse(y == 1, 1, -1)

      # check no NA - anyNA() is by far fastest solution
      if(anyNA(X@x))
        stop("NA's in input matrix are not allowed")

      p = fm_partial_fit(private$ptr_param, X, y, do_update = TRUE, nthread = nthread)
      invisible(p)
    },
    predict =  function(X, nthread = 0, ...) {

      stopifnot(private$is_initialized)
      if(!inherits(class(X), private$internal_matrix_format)) {
        # message(Sys.time(), " casting input matrix (class ", class(X), ") to ", private$internal_matrix_format)
        X = as(X, private$internal_matrix_format)
      }
      stopifnot(ncol(X) == private$model$n_features)

      if(any(is.na(X)))
        stop("NA's in input matrix are not allowed")

      p = fm_partial_fit(private$ptr_param, X, numeric(0), do_update = FALSE, nthread = nthread)
      return(p);
    },
    dump = function() {
      fm_dump(private$ptr_param)
    }
  ),
  private = list(
    #--------------------------------------------------------------
    is_initialized = FALSE,
    internal_matrix_format = "RsparseMatrix",
    #--------------------------------------------------------------
    ptr_param = NULL,
    ptr_model = NULL,
    #--------------------------------------------------------------
    n_features = NULL,
    learning_rate = NULL,
    rank = NULL,
    lambda_w = NULL,
    lambda_v = NULL,
    family = NULL,
    family_name = NULL,
    #--------------------------------------------------------------
    init_model_param = function(learning_rate, rank, lambda_w, lambda_v, family) {

      private$learning_rate = learning_rate
      private$rank = rank
      private$lambda_w = lambda_w
      private$lambda_v = lambda_v
      private$family = family

      private$ptr_param = fm_create_param(learning_rate, rank, lambda_w, lambda_v)
    },

    init_model_state = function(n_features, w = NULL, v = NULL) {
      # init R side model params
      private$n_features = n_features

      v_init_stddev = 0.001
      v_init_mean = 0

      # init with 0
      if(is.null(w))
        w = runif(n_features, v_init_mean, v_init_stddev)
      # init with small numbers
      if(is.null(v))
        v = matrix(runif(n_features * private$rank, v_init_mean, v_init_stddev),
                   nrow = n_features, ncol = private$rank)
      # update param pointer
      fm_init_weights(private$ptr_param, w, v)
      rm(w, v); gc();
      # mark that we initialized model
      private$is_initialized = TRUE
    }
  )
)

