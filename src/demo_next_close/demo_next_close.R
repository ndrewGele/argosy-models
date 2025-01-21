demo_next_close <- function(
  x.data, features, y.function,
  db.con, cutoff.date,
  tune.initial, tune.iter, tune.no.improve
) {
  
  require(dplyr)

  require(rsample)
  require(recipes)
  require(parsnip)
  require(workflows)
  require(workflowsets)
  require(yardstick)
  require(tune)
  
  require(glmnet)
  require(ranger)
  
  # Data setup (Y Var, Test/Train Split)
  y_df <- y.function(db.con = db.con)
  
  df <- x.data %>% 
    bind_cols(features) %>% 
    inner_join(
      y_df,
      by = c('symbol', 'date')
    ) %>% 
    # Removing Inf and NA values because Recipes wasn't doing it???
    filter_all(all_vars(!is.infinite(.))) %>% 
    filter_all(all_vars(!is.na(.)))
  
  train_df <- filter(df, date < cutoff.date)
  test_df <- filter(df, date >= cutoff.date)
  
  num_months <- lubridate::floor_date(
    train_df$date, 
    unit = 'months'
  ) %>% 
    unique() %>% 
    length()
  
  lookback_skip <- ceiling(num_months * .2)
  assess_length <- max( floor((num_months - 4) / 5), 1)
  
  # Create folds for model tuning
  random_folds <- rsample::vfold_cv(train_df, v = 5)
  time_folds <- rsample::sliding_period(
    data = train_df %>% arrange(date),
    index = date,
    period = 'month',
    lookback = Inf,
    assess_start = 1,
    assess_stop = assess_length,
    complete = FALSE,
    skip = lookback_skip
  ) %>% 
    tail(5)
  
  all_folds <- rsample::manual_rset(
    splits = c(random_folds$splits, time_folds$splits),
    ids = c(random_folds$id, time_folds$id)
  )
  
  # Initialize recipes, model specs, workflowset
  base_recipe <- recipes::recipe(
    demo_next_close ~ .,
    data = train_df
  ) %>%
    recipes::update_role(symbol, new_role = 'id') %>%
    recipes::update_role(date, new_role = 'time') %>%
    recipes::step_date(date, features = c('dow', 'doy')) %>%
    recipes::step_dummy(all_nominal_predictors()) %>%
    recipes::step_zv(all_predictors())
  
  normalized_recipe <- base_recipe %>%
    recipes::step_normalize(all_numeric_predictors())
  
  glmnet_spec <- parsnip::linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    parsnip::set_engine('glmnet') %>%
    parsnip::set_mode('regression')
  
  rand_forest_spec <- rand_forest(trees = 25, min_n = tune()) %>%
    parsnip::set_engine('ranger') %>%
    parsnip::set_mode('regression')
  
  wf_set <- workflowsets::workflow_set(
    preproc = list(
      base = base_recipe,
      normalized = normalized_recipe
    ),
    models =  list(
      glm = glmnet_spec,
      rf = rand_forest_spec
    ),
    cross = TRUE
  )
  
  
  # Map over workflows in set
  wf_set_tuned <- workflowsets::workflow_map(
    object = wf_set,
    fn = 'tune_bayes',
    verbose = TRUE,
    resamples = all_folds,
    metrics = yardstick::metric_set(yardstick::mae),
    objective = tune::exp_improve(),
    initial = tune.initial,
    iter = tune.iter,
    control = tune::control_bayes(
      no_improve = tune.no.improve,
      verbose = TRUE
    )
  )
  
  results <- purrr::map(
    .x = wf_set_tuned$wflow_id,
    .f = function(x) {
      
      best_results <- wf_set_tuned %>%
        workflowsets::extract_workflow_set_result(x) %>%
        tune::select_best(metric = 'mae')
      
      best_wflow <- wf_set_tuned %>%
        workflowsets::extract_workflow(x) %>%
        tune::finalize_workflow(best_results) %>%
        generics::fit(data = train_df)
      
      training_predicted <- train_df %>%
        bind_cols(predict(best_wflow, .)) %>%
        mae(
          truth = demo_next_close,
          estimate = .pred
        )
      
      holdout_predicted <- test_df %>%
        bind_cols(predict(best_wflow, .)) %>%
        mae(
          truth = demo_next_close,
          estimate = .pred
        )
      
      result <- list()
      result$model_obj <- best_wflow
      result$model_name <- 'demo_next_close'
      result$model_recipe <- strsplit(x, '_')[[1]][1]
      result$model_model <- strsplit(x, '_')[[1]][2]
      result$model_obj_type <- 'workflow'
      result$model_create_date <- Sys.Date()
      result$feature_hash <- substr(rlang::hash(sort(names(features))), 1, 8)
      result$model_hash <- substr(rlang::hash(best_wflow), 1, 8)
      result$model_training_perf <- training_predicted$.estimate
      result$model_holdout_perf <- holdout_predicted$.estimate
      
      return(result)
      
    }
  )
  
  return(results)
  
}
