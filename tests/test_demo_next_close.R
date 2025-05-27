# Test Case
library(dplyr)
library(dbplyr)


# Connect to DB -----------------------------------------------------------

db_con <- DBI::dbConnect(
  drv = RPostgres::Postgres(),
  dbname = Sys.getenv('POSTGRES_DB'),
  host = Sys.getenv('POSTGRES_HOST'),
  port = Sys.getenv('POSTGRES_PORT'),
  user = Sys.getenv('POSTGRES_USER'),
  password = Sys.getenv('POSTGRES_PASSWORD')
)


# Pull in X Var Data ------------------------------------------------------

source(glue::glue('{Sys.getenv("COMMON_CODE_CONTAINER")}/src/get_all_data.R'))

symbols <- db_con |>
  tbl('symbols') |>
  inner_join(
    db_con |>
      tbl('symbols') |>
      group_by(fetcher_name) |>
      summarise(update_timestamp = max(update_timestamp, na.rm = TRUE)),
    by = c('fetcher_name', 'update_timestamp')
  ) |>
  select(symbol) |>
  distinct() |>
  collect() |>
  pull(symbol)

x_data <- get_all_data(
  db.con = db_con,
  symbols = symbols,
  start.date = lubridate::ymd('2022-01-01'),
  end.date = lubridate::ymd('2023-12-31')
)


# Create model features ---------------------------------------------------

source(glue::glue('{Sys.getenv("COMMON_CODE_CONTAINER")}/src/feature_utils.R'))
feature_def <- source('./src/demo_next_close/demo_next_close_features.R')$value
feature_spec <- generate_feature_spec(feature_def)
features_df <- create_features(x_data, feature_spec)

feature_hash <- feature_spec |> 
  names() |> 
  sort() |>
  rlang::hash() |> 
  substr(1, 8)

feature_spec_string <- feature_spec |> 
  as.character() |> 
  jsonlite::toJSON() |> 
  as.character()


# Pull Y Data -------------------------------------------------------------

y_function <- source('./src/demo_next_close/demo_next_close_y.R')$value
y_data <- y_function(db.con = db.con)


# Combine Data ------------------------------------------------------------

df <- x_data |> 
  bind_cols(features) |> 
  inner_join(
    y.data,
    by = c('symbol', 'date')
  )
  

# Tune and Train Model ----------------------------------------------------

model_fun <- source('./src/demo_next_close/demo_next_close.R')$value

test_model_results <- model_fun(
  data = df,
  cutoff.date = lubridate::ymd(Sys.getenv('MODEL_CUTOFF')),
  tune.initial = as.integer(Sys.getenv('MODEL_TUNE_INITIAL')),
  tune.iter = as.integer(Sys.getenv('MODEL_TUNE_ITER')),
  tune.no.improve = as.integer(Sys.getenv('MODEL_TUNE_NO_IMPROVE'))
)


# Simulate Database Write -------------------------------------------------

feature_table <- data.frame(
  feature_hash = feature_hash,
  feature_spec_string = feature_spec_string
)

test_model_results_table <- purrr::map_dfr(
  .x = test_model_results,
  .f = function(x) {
    data.frame(
      name = x$model_name,
      recipe = x$model_recipe,
      model = x$model_model,
      feature_hash = feature_hash,
      training_perf = x$model_training_perf,
      holdout_perf = x$model_holdout_perf,
      file_name = glue::glue('{x$model_name}_{x$feature_hash}.RDS'),
      mode = 'new', # this is picked mode
      update_timestamp = Sys.time()
    )
  }
)


# Tweak Features and Re-run Model -----------------------------------------

# Just pick one model to use in this example
sample_model <- test_model_results_df |> 
  sample_n(1)

# We can take the feature spec json and convert it back to a list for tweaking
saved_feature_spec <- feature_table$feature_spec_string |> 
  jsonlite::fromJSON() |> 
  purrr::map(\(x) eval(parse(text = x)))

tweaked_feature_spec <- tweak_feature_spec(
  feature.spec = saved_feature_spec,
  feature.def = source('./src/demo_next_close/demo_next_close_features.R')$value
)

tweaked_feature_hash <- tweaked_feature_spec |> 
  names() |> 
  sort() |>
  rlang::hash() |> 
  substr(1, 8)

tweaked_feature_spec_string <- tweaked_feature_spec |> 
  as.character() |> 
  jsonlite::toJSON() |> 
  as.character()

test_tweaked_features <- create_features(
  data = x_data,
  feature.spec = test_tweaked_spec
)

test_tweaked_model_results <- model_fun(
  x.data = x_data,
  features = test_tweaked_features,
  y.data = y_data,
  cutoff.date = lubridate::ymd(Sys.getenv('MODEL_CUTOFF')),
  tune.initial = as.integer(Sys.getenv('MODEL_TUNE_INITIAL')),
  tune.iter = as.integer(Sys.getenv('MODEL_TUNE_ITER')),
  tune.no.improve = as.integer(Sys.getenv('MODEL_TUNE_NO_IMPROVE'))
)

DBI::dbDisconnect(db_con)


# Simulate Tweaked Database Write -----------------------------------------

# Features lookup table will map hashes to spec strings
feature_table <- feature_table |> 
  bind_rows(
    data.frame(
      feature_hash = tweaked_feature_hash,
      feature_spec_string = tweaked_feature_spec_string
    )
  )

test_tweaked_model_results_df <- purrr::map_dfr(
  .x = test_tweaked_model_results,
  .f = function(x) {
    data.frame(
      name = x$model_name,
      recipe = x$model_recipe,
      model = x$model_model,
      feature_hash = tweaked_feature_hash,
      training_perf = x$model_training_perf,
      holdout_perf = x$model_holdout_perf,
      file_name = glue::glue('{x$model_name}_{x$feature_hash}.RDS'),
      mode = 'tweaked',
      update_timestamp = Sys.time()
    )
  }
)

# Now we can look at a larger set of results using two feature sets
all_results <- bind_rows(
  test_model_results_df,
  test_tweaked_model_results_df
)
