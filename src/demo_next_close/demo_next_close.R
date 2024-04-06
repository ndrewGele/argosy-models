demo_next_close <- function(
  x.data, y.function,
  db.con, cutoff.date,
  tune.initial, tune.iter, tune.no.improve
) {
  
  require(dplyr)
  require(lubridate)
  
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
  
  df <- inner_join(
    x.data,
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
  random_folds <- vfold_cv(train_df, v = 5)
  time_folds <- sliding_period(
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
  
  all_folds <- manual_rset(
    splits = c(random_folds$splits, time_folds$splits),
    ids = c(random_folds$id, time_folds$id)
  )
  
  # Initialize recipes, model specs, workflowset
  base_recipe <- recipe(
    demo_next_close ~ .,
    data = train_df
  ) %>%
    update_role(symbol, new_role = 'id') %>%
    update_role(date, new_role = 'time') %>%
    step_date(date, features = c('dow', 'doy')) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())
  
  normalized_recipe <- base_recipe %>%
    step_normalize(all_numeric_predictors())
  
  glmnet_spec <- linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine('glmnet') %>%
    set_mode('regression')
  
  rand_forest_spec <- rand_forest(trees = 25, min_n = tune()) %>%
    set_engine('ranger') %>%
    set_mode('regression')
  
  wf_set <- workflow_set(
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
  wf_set_tuned <- workflow_map(
    object = wf_set,
    fn = 'tune_bayes',
    verbose = TRUE,
    resamples = all_folds,
    metrics = metric_set(mae),
    objective = exp_improve(),
    initial = tune.initial,
    iter = tune.iter,
    control = control_bayes(
      no_improve = tune.no.improve,
      verbose = TRUE
    )
  )
  
  results <- purrr::map(
    .x = wf_set_tuned$wflow_id,
    .f = function(x) {
      
      best_results <- wf_set_tuned %>%
        extract_workflow_set_result(x) %>%
        select_best(metric = 'mae')
      
      best_wflow <- wf_set_tuned %>%
        extract_workflow(x) %>%
        finalize_workflow(best_results) %>%
        fit(data = train_df)
      
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
      result$model_hash <- substr(rlang::hash(best_wflow), 1, 8)
      result$model_training_perf <- training_predicted$.estimate
      result$model_holdout_perf <- holdout_predicted$.estimate
      
      return(result)
      
    }
  )
  
  return(results)
  
}


# # Test Case
# db_con <- DBI::dbConnect(
#   drv = RPostgres::Postgres(),
#   dbname = Sys.getenv('POSTGRES_DB'),
#   host = Sys.getenv('POSTGRES_HOST'),
#   port = Sys.getenv('POSTGRES_PORT'),
#   user = Sys.getenv('POSTGRES_USER'),
#   password = Sys.getenv('POSTGRES_PASSWORD')
# )
# 
# source(glue::glue('{Sys.getenv("COMMON_CODE_CONTAINER")}/src/get_all_data.R'))
# x_data <- get_all_data(
#   db.con = db_con,
#   start.date = lubridate::ymd('2022-01-01'),
#   end.date = lubridate::ymd('2023-12-31')
# )
# 
# cutoff <- lubridate::ymd(Sys.getenv('MODEL_CUTOFF'))
# 
# test_model_results <- demo_next_close(
#   x.data = x_data,
#   db.con = db_con,
#   cutoff.date = cutoff,
#   tune.initial = 4,
#   tune.iter = 5,
#   tune.no.improve = 3
# )
# 
# DBI::dbDisconnect(db_con)
