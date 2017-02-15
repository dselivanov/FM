#' @useDynLib FM
#' @importFrom Rcpp evalCpp
NULL

#' @name FM
#' @title Creates FactorizationMachines model.
#' @description Creates second order Factorization Machines model
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' fm = FM$new(learning_rate = 0.2, rank = 8, lambda_w = 1e-6, lambda_v = 1e-6, task = "classification")
#' fm$partial_fit(X, y, nthread  = 0, ...)
#' fm$predict(X, nthread  = 0, ...)
#' }
#' @format \code{\link{R6Class}} object.
#' @section Methods:
#' \describe{
#'   \item{\code{FM$new(learning_rate = 0.2, rank = 8, lambda_w = 1e-6, lambda_v = 1e-6, task = "classification")}}{Constructor
#'   for FactorizationMachines model. For description of arguments see \bold{Arguments} section.}
#'   \item{\code{$partial_fit(X, y, nthread  = 0, ...)}}{fits/updates model given input matrix \code{X} and target vector \code{y}.
#'   \code{X} shape = (n_samples, n_features)}
#'   \item{\code{$predict(X, nthread  = 0, ...)}}{predicts output \code{X}}
#'   \item{\code{$coef()}}{ return coefficients of the regression model}
#'   \item{\code{$dump()}}{create dump of the model (actually \code{list}) with current model parameters}
#'}
#' @section Arguments:
#' \describe{
#'  \item{fm}{\code{FTRL} object}
#'  \item{X}{Input sparse matrix - native format is \code{Matrix::RsparseMatrix}.
#'  If \code{X} is in different format, model will try to convert it to \code{RsparseMatrix}
#'  with \code{as(X, "RsparseMatrix")} call}
#'  \item{learning_rate}{learning rate for AdaGrad SGD}
#'  \item{rank}{rank of the latent dimension in factorization}
#'  \item{lambda_w}{regularization parameter for linear terms}
#'  \item{lambda_v}{regularization parameter for interactions terms}
#'  \item{n_features}{number of features in model (number of columns in expected model matrix) }
#'  \item{task}{ \code{"regression"} or \code{"classification"}}
#' }
#' @export
FM = R6::R6Class(
  classname = "estimator",
  public = list(
    #-----------------------------------------------------------------
    initialize = function(learning_rate = 0.2, rank = 4,
                          lambda_w = 0, lambda_v = 0,
                          task = c("classification", "regression")) {
      stopifnot(lambda_w >= 0 && lambda_v >= 0 && learning_rate > 0 && rank >= 1)
      task = match.arg(task);
      private$init_model_param(learning_rate = learning_rate, rank = rank,
                               lambda_w = lambda_w, lambda_v = lambda_v,
                               task = task)
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
      if(private$task == 'classification')
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
    task = NULL,
    #--------------------------------------------------------------
    init_model_param = function(learning_rate, rank, lambda_w, lambda_v, task) {

      private$learning_rate = learning_rate
      private$rank = rank
      private$lambda_w = lambda_w
      private$lambda_v = lambda_v
      private$task = task

      private$ptr_param = fm_create_param(learning_rate, rank, lambda_w, lambda_v, task)
    },

    init_model_state = function(n_features, w = NULL, v = NULL) {
      # init R side model params
      private$n_features = n_features

      v_init_stddev = 0.001
      v_init_mean = 0

      # init with 0
      if(is.null(w))
        w = numeric(n_features)
      # init with small numbers
      if(is.null(v))
        v = matrix(rnorm(n_features * private$rank, v_init_mean, v_init_stddev),
                   nrow = n_features, ncol = private$rank)
      # update param pointer
      fm_init_weights(private$ptr_param, w, v)
      rm(w, v); gc();
      # mark that we initialized model
      private$is_initialized = TRUE
    }
  )
)

