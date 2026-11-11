# Parity fixtures for chapkit-ts' ARIMA from R itself (`make arima-fixtures` runs both):
#
#   Rscript scripts/gen_arima_fixtures.R stats   # stats::arima, any R >= 4.0 (made with 4.6.2)
#   Rscript scripts/gen_arima_fixtures.R fable   # fable::ARIMA's search, in an image with fable 0.5.0
#
# The fable fixtures were made in ghcr.io/chap-models/auto_arima_chapkit (fable 0.5.0, feasts
# 0.5.0, R 4.5.0, linux/amd64), from the repository root mounted at /w.
#
# Writes crates/chapkit-ts/fixtures/{stats,fable}.json. The series come from fixed seeds.

js_num <- function(x) {
  if (length(x) == 0) return("[]")
  v <- ifelse(is.na(x), "null", ifelse(is.infinite(x), ifelse(x > 0, "1e999", "-1e999"), sprintf("%.17g", x)))
  paste0("[", paste(v, collapse = ","), "]")
}
js <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x))) {
      return(paste0("{", paste(sprintf("\"%s\":%s", names(x), vapply(x, js, "")), collapse = ","), "}"))
    }
    return(paste0("[", paste(vapply(x, js, ""), collapse = ","), "]"))
  }
  if (is.character(x)) return(if (length(x) == 1) sprintf("\"%s\"", x) else paste0("[", paste(sprintf("\"%s\"", x), collapse = ","), "]"))
  if (is.logical(x) && length(x) == 1) return(if (x) "true" else "false")
  if (length(x) == 1 && !is.null(attr(x, "scalar"))) return(if (is.na(x)) "null" else sprintf("%.17g", x))
  js_num(as.numeric(x))
}
scalar <- function(x) structure(x, scalar = TRUE)

series <- function(seed, n, period, level, amp, ar, noise, trend = 0) {
  set.seed(seed)
  e <- as.numeric(arima.sim(list(ar = ar), n = n, sd = noise))
  t <- seq_len(n)
  level + trend * t + amp * sin(2 * pi * t / period) + e
}

arima_constant <- function(n, d, D, period) {
  constant <- rep(1, n)
  if (d > 0) constant <- stats::diffinv(constant, differences = d, xi = rep(1, d))[seq_len(n)]
  if (D > 0) constant <- stats::diffinv(constant, lag = period, differences = D, xi = rep(1, period * D))[seq_len(n)]
  constant
}

stats_case <- function(name, y, order, seasonal, constant, method, h = 8) {
  period <- seasonal[4]
  xreg <- NULL
  newxreg <- NULL
  if (constant) {
    cc <- arima_constant(length(y) + h, order[2], seasonal[2], period)
    xreg <- matrix(cc[seq_along(y)], dimnames = list(NULL, "constant"))
    newxreg <- matrix(cc[length(y) + seq_len(h)], dimnames = list(NULL, "constant"))
  }
  fit <- tryCatch(arima(ts(y, frequency = period), order = order, seasonal = list(order = seasonal[1:3], period = period),
               xreg = xreg, include.mean = FALSE, method = method), error = function(e) NULL)
  if (is.null(fit)) return(list(name = name, y = y, order = c(order, seasonal), constant = constant, method = method, error = TRUE))
  fc <- predict(fit, n.ahead = h, newxreg = newxreg)
  fixed <- arima(ts(y, frequency = period), order = order, seasonal = list(order = seasonal[1:3], period = period),
                 xreg = xreg, include.mean = FALSE, method = method, fixed = coef(fit))
  list(name = name, y = y, order = c(order, seasonal), constant = constant, method = method, error = FALSE,
       coef = unname(coef(fit)), sigma2 = scalar(fit$sigma2), loglik = scalar(fit$loglik),
       aic = scalar(if (is.na(fit$aic)) NA_real_ else fit$aic), residuals = as.numeric(residuals(fit)),
       phi = fit$model$phi, theta = fit$model$theta,
       mean = as.numeric(fc$pred), se = as.numeric(fc$se),
       fixed_sigma2 = scalar(fixed$sigma2), fixed_loglik = scalar(fixed$loglik))
}

stats_fixtures <- function() {
  m1 <- series(1, 72, 12, 50, 15, c(0.5), 4)
  m2 <- series(2, 60, 12, 20, 6, c(0.3, 0.2), 2, trend = 0.15)
  w1 <- series(3, 160, 52, 100, 30, c(0.6), 8)
  m3 <- m1
  m3[c(10, 40)] <- NA
  cases <- list(
    stats_case("arma11_c", m1, c(1, 0, 1), c(0, 0, 0, 12), TRUE, "CSS-ML"),
    stats_case("sarima_211_011", m2, c(2, 1, 1), c(0, 1, 1, 12), FALSE, "CSS-ML"),
    stats_case("drift_011_100", m2, c(0, 1, 1), c(1, 0, 0, 12), TRUE, "CSS-ML"),
    stats_case("seasonal_ar_ma", m1, c(1, 0, 0), c(1, 0, 1, 12), TRUE, "CSS-ML"),
    stats_case("css", m1, c(2, 0, 1), c(1, 0, 0, 12), TRUE, "CSS"),
    stats_case("ml", m2, c(1, 1, 0), c(0, 0, 1, 12), FALSE, "ML"),
    stats_case("missing", m3, c(1, 0, 1), c(0, 0, 0, 12), TRUE, "CSS-ML"),
    stats_case("weekly_100_100", w1, c(1, 0, 0), c(1, 0, 0, 52), TRUE, "CSS-ML"),
    stats_case("weekly_css", w1, c(2, 0, 0), c(1, 0, 1, 52), TRUE, "CSS"),
    stats_case("weekly_011_010", w1, c(0, 1, 1), c(0, 1, 0, 52), FALSE, "CSS-ML")
  )
  cat(js(list(r = R.version.string, cases = cases)), "\n", file = "crates/chapkit-ts/fixtures/stats.json")
}

fable_fixtures <- function() {
  suppressPackageStartupMessages({
    library(fable)
    library(tsibble)
    library(dplyr)
  })
  one <- function(name, y, period, h = 6, extra = 12) {
    idx <- if (period == 12) yearmonth("2015 Jan") + seq_along(y) - 1 else yearweek("2015 W01") + seq_along(y) - 1
    full <- tsibble(t = idx, y = y, index = t)
    n <- length(y) - extra
    train <- full[seq_len(n), ]
    t0 <- Sys.time()
    fit <- model(train, a = ARIMA(y))
    secs <- as.numeric(Sys.time() - t0, units = "secs")
    m <- fit$a[[1]]$fit
    spec <- m$spec
    fc <- forecast(fit, h = h)
    dist <- fc$y
    later <- refit(fit, full, reestimate = FALSE)
    lfc <- forecast(later, h = h)
    lm <- later$a[[1]]$fit
    list(name = name, period = period, y = y, n_train = n,
         order = as.numeric(c(spec$p, spec$d, spec$q, spec$P, spec$D, spec$Q)), constant = isTRUE(spec$constant),
         coef = unname(m$model$coef), sigma2 = scalar(m$fit$sigma2), aicc = scalar(m$fit$AICc), loglik = scalar(m$fit$log_lik),
         mean = as.numeric(mean(dist)), sd = sqrt(as.numeric(distributional::variance(dist))),
         refit_sigma2 = scalar(lm$fit$sigma2),
         refit_mean = as.numeric(mean(lfc$y)), refit_sd = sqrt(as.numeric(distributional::variance(lfc$y))),
         seconds = scalar(secs))
  }
  set.seed(10)
  cases <- list(
    one("monthly_seasonal", series(11, 84, 12, 60, 25, c(0.5), 6), 12),
    one("monthly_counts", round(pmax(0, series(12, 72, 12, 30, 20, c(0.7), 5))), 12),
    one("monthly_trend", series(13, 96, 12, 40, 10, c(0.4, 0.2), 3, trend = 0.3), 12),
    one("monthly_short", series(14, 36, 12, 10, 4, c(0.2), 1.5), 12),
    one("monthly_noise", series(15, 60, 12, 5, 0.5, c(0.1), 2), 12),
    one("monthly_walk", 50 + cumsum(series(16, 80, 12, 0, 1, c(0.3), 2)), 12),
    one("weekly_seasonal", series(21, 170, 52, 100, 40, c(0.6), 10), 52),
    one("weekly_counts", round(pmax(0, series(22, 220, 52, 40, 30, c(0.8), 6))), 52),
    one("weekly_short", series(23, 120, 52, 20, 8, c(0.5), 3), 52),
    # Weekly dengue cases in Matagalpa, Nicaragua (Chap's example data): the first 242 weeks pick
    # ARIMA(3,1,1)(1,0,1)[52], whose initial covariance Gardner's method cannot solve accurately.
    one("weekly_matagalpa", scan("crates/chapkit-ts/fixtures/matagalpa_weekly.txt", quiet = TRUE), 52, extra = 9)
  )
  # Refits on a history shorter than the one trained on (P, Q above a third of it in periods).
  refit_case <- function(name, y, n_refit, h = 3) {
    full <- tsibble(t = yearmonth("2010 Jan") + seq_along(y) - 1, y = y, index = t)
    fit <- model(full, a = ARIMA(y))
    m <- fit$a[[1]]$fit
    out <- list(name = name, y = y, n_refit = n_refit,
                order = as.numeric(c(m$spec$p, m$spec$d, m$spec$q, m$spec$P, m$spec$D, m$spec$Q)),
                constant = isTRUE(m$spec$constant), coef = unname(m$model$coef))
    for (re in c(FALSE, TRUE)) {
      key <- if (re) "reestimate" else "filter"
      r <- tryCatch(refit(fit, full[seq_len(n_refit), ], reestimate = re), error = function(e) NULL)
      if (is.null(r)) {
        out[[key]] <- list(error = TRUE)
        next
      }
      d <- forecast(r, h = h)$y
      out[[key]] <- list(error = FALSE, coef = unname(r$a[[1]]$fit$model$coef), sigma2 = scalar(r$a[[1]]$fit$fit$sigma2),
                         mean = as.numeric(mean(d)), sd = sqrt(as.numeric(distributional::variance(d))))
    }
    out
  }
  refits <- list(
    # Dengue cases in Bokeo, Laos (Chap's laos_subset example data), 2010-2012.
    refit_case("bokeo_monthly", scan("crates/chapkit-ts/fixtures/bokeo_monthly.txt", quiet = TRUE), 20)
  )
  data_nrow <- NROW(get("data", envir = asNamespace("fable")))
  cat(js(list(r = R.version.string, fable = as.character(packageVersion("fable")),
              feasts = as.character(packageVersion("feasts")), nrow_data = scalar(data_nrow), cases = cases,
              refits = refits)), "\n",
      file = "crates/chapkit-ts/fixtures/fable.json")
}

what <- commandArgs(trailingOnly = TRUE)[1]
if (identical(what, "stats")) stats_fixtures() else if (identical(what, "fable")) fable_fixtures() else stop("stats or fable")
