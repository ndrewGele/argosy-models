demo_next_close_y <- function(db.con) {

  require(dplyr)
  require(dbplyr)

  symbols <- db_con %>%
    tbl('symbols') %>%
    inner_join(
      db_con %>%
        tbl('symbols') %>%
        group_by(fetcher_name) %>%
        summarise(update_timestamp = max(update_timestamp, na.rm = TRUE)),
      by = c('fetcher_name', 'update_timestamp')
    ) %>%
    select(symbol) %>%
    distinct() %>%
    collect() %>%
    pull(symbol)

  res <- db.con %>%
    tbl('daily_ohlc') %>%
    filter(symbol %in% !!symbols) %>%
    select(symbol, timestamp, high, close) %>%
    arrange(symbol, timestamp) %>% 
    collect() %>%
    group_by(symbol) %>%
    mutate(demo_next_close = dplyr::lead(close)) %>% 
    mutate(demo_next_close = demo_next_close/close) %>% 
    filter(!is.na(demo_next_close)) %>% 
    mutate(date = lubridate::floor_date(timestamp, 'day')) %>% 
    select(
      symbol,
      date,
      demo_next_close
    )

  return(res)

}
