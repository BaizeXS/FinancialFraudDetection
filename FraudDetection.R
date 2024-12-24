# ============================================
# 0. Prepare Environment and Load Packages
# ============================================
##############################
# Install and Load Packages #
#############################
options(repos = c(CRAN = "https://cran.rstudio.com/"))

if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  # Utilities
  conflicted, DescTools, doParallel, logger,
  # Core Data Manipulation
  tidyverse, dplyr, readr,
  # Machine Learning & Modeling
  caret, glmnet, randomForest, ranger, xgboost,
  lightgbm, e1071, nnet, neuralnet,
  # Model Evaluation & Metrics
  pROC, MLmetrics,
  # Imbalanced Data Handling
  smotefamily, ROSE,
  # Visualization
  ggplot2, corrplot, purrr
)



############################
# Configure the log system #
############################
# Set up the log directory and file path
if (!dir.exists("./logs")) {
  dir.create("./logs", recursive = TRUE)
}
LOG_FILE <- "./logs/fraud_detection.log"

# Helper function: Add line breaks and alignment to long messages
format_message <- function(msg, width = 100, timestamp_width = 28) {
  # Split message into lines
  lines <- strwrap(msg, width = width - timestamp_width)
  # The first line does not need alignment, the subsequent lines need alignment
  if (length(lines) > 1) {
    # Add alignment spaces to the subsequent lines
    lines[-1] <- paste(
      paste(rep(" ", timestamp_width), collapse = ""),
      lines[-1]
    )
  }
  # Combine all lines
  paste(lines, collapse = "\n")
}

# Set up colorful log layout
color_layout <- layout_glue_generator(
  format = paste0(
    # Log level color
    "{if (level == 'ERROR') '\033[1;31m' else", # Bold red
    " if (level == 'WARN') '\033[1;33m' else", # Bold yellow
    " if (level == 'INFO') '\033[1;32m' else", # Bold green
    " if (level == 'DEBUG') '\033[1;36m'", # Bold cyan
    " else '\033[1;34m'}", # Bold blue (default)
    "{sprintf('%-5s', level)}", # Fixed width log level
    "\033[0m", # Reset color
    " [",
    # Timestamp
    "\033[90m{format(time, '%Y-%m-%d %H:%M:%S')}\033[0m",
    "] ",
    # Message color
    "{if (level == 'ERROR') '\033[31m' else", # Red message
    " if (level == 'WARN') '\033[33m' else", # Yellow message
    " if (level == 'INFO') '\033[32m' else", # Green message
    " if (level == 'DEBUG') '\033[36m'", # Cyan message
    " else '\033[34m'}", # Blue message (default)
    "{format_message(msg)}\033[0m" # Format message and reset color
  )
)

# Set up the log system
log_appender(appender_tee(LOG_FILE))
log_layout(color_layout)
log_threshold(TRACE)

# Test the log system
log_info("=== Fraud Detection System Initialized ===")
log_info(sprintf("Log file location: %s", LOG_FILE))



#################################
# Configure parallel processing #
#################################
# Initialize parallel cluster
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
log_info("Initialized parallel cluster with {detectCores() - 1} cores.")



# ============================================
# 1. Utility Functions
# ============================================

# Get the mode of a vector
getmode <- function(v) {
  if (all(is.na(v))) {
    log_warn("Input vector is all NA")
    return(NA)
  }

  v_clean <- na.omit(v)
  if (length(v_clean) == 0) {
    log_warn("Vector is empty after removing NA")
    return(NA)
  }

  mode_val <- unique(v_clean)[which.max(tabulate(match(v_clean, unique(v_clean))))] # nolint: line_length_linter.
  return(mode_val)
}

# Get the type of each variable
get_variable_types <- function(data) {
  # Get the type of each variable
  variable_types <- sapply(data, class)

  # Classify variables by type
  variables_by_type <- split(names(data), variable_types)

  # Optimize output
  log_info("\n=== Data Variable Types Summary ===")
  for (type in names(variables_by_type)) {
    # Construct each type of variable information
    vars_text <- paste(
      strwrap(
        paste(variables_by_type[[type]], collapse = ", "),
        width = 60
      ),
      collapse = "\n"
    )

    # Print each type of variable information
    log_info(sprintf("Type: %s", type))
    log_info(sprintf("Variables:\n%s", vars_text))
  }
}

# Create a header with a decorative border
create_header <- function(text, width = 78) {
  stopifnot(nchar(text) <= width - 4)

  # Generate top and bottom borders
  top <- paste0("┌", strrep("─", width), "┐")
  bottom <- paste0("└", strrep("─", width), "┘")

  # Calculate left and right padding to center the text
  total_padding <- width - 2 - nchar(text)
  left_padding <- floor(total_padding / 2)
  right_padding <- total_padding - left_padding

  # Build content line
  content <- paste0(
    "│",
    strrep(" ", left_padding),
    text,
    strrep(" ", right_padding),
    "│"
  )

  # Return the header with three lines
  c(top, content, bottom)
}

# Wrapper function for logging headers
log_header <- function(text, width = 78) {
  for (line in create_header(text, width)) {
    log_info(line)
  }
}

# Generalized plot saving function
save_plot <- function(plot, output_dir, filename, width = 10, height = 6, dpi = 300) { # nolint: line_length_linter.
  ggsave(
    filename = file.path(output_dir, paste0(filename, ".png")),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi
  )
}

# Generalized plot creation and saving function
create_and_save_plot <- function(
    data,
    aes_mapping,
    geom_func,
    geom_args = list(),
    labs_args = list(),
    theme_args = list(),
    output_dir,
    filename,
    fill_color = NULL) {
  p <- ggplot(data, do.call(aes, aes_mapping)) +
    do.call(geom_func, c(list(), geom_args)) +
    labs(do.call(labs, labs_args)) +
    theme_minimal() +
    theme(do.call(theme, theme_args))

  if (!is.null(fill_color)) {
    p <- p + scale_fill_manual(values = fill_color)
  }

  save_plot(p, output_dir, filename)
}

# ============================================
# 2. EDA
# ============================================

# 2.1 Print basic information
basic_information <- function(data) {
  log_info("Basic Information")
  log_info(sprintf("Dimensions: %d rows × %d columns", nrow(data), ncol(data))) # nolint: line_length_linter.

  # Target distribution
  target_dist <- table(data$isFraud)
  log_info(sprintf(
    "Target Distribution: Non-fraud: %d (%.1f%%), Fraud: %d (%.1f%%)",
    target_dist[1], 100 * target_dist[1] / sum(target_dist),
    target_dist[2], 100 * target_dist[2] / sum(target_dist)
  ))

  # Variable types
  var_types <- table(sapply(data, class))
  log_info("Variable Types:")
  for (i in seq_along(var_types)) {
    log_info(sprintf("  %s: %d", names(var_types)[i], var_types[i]))
  }
}

# 2.2 Missing values analysis and handling
# Helper Function: Convert empty strings to NA
convert_empty_strings_to_na <- function(data) {
  character_cols <- sapply(data, is.character)
  for (column in names(data)[character_cols]) {
    data[[column]][data[[column]] == ""] <- NA
  }
  return(data)
}

# Helper Function: Remove columns with high missing values
remove_high_missing_columns <- function(data, missing_percentage, remove_threshold = 90) { # nolint: line_length_linter.
  # Find columns with missing values greater than remove_threshold
  high_missing_cols <- names(which(missing_percentage >= remove_threshold))

  # Remove columns with missing values greater than remove_threshold
  if (length(high_missing_cols) > 0) {
    log_info(sprintf(
      "Removing columns with missing values greater than %d%%: %s",
      remove_threshold, paste(high_missing_cols, collapse = ", ")
    ))
    data <- data[, !(names(data) %in% high_missing_cols)]
  }

  return(data)
}

# Helper function for filling missing values
fill_missing_values <- function(data) {
  fill_unknown_cols <- c(
    "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "id_30", "id_34", "id_31", "id_33",
    "DeviceInfo", "M4"
  )
  fill_mode_cols <- c("id_15", "id_16", "id_28", "id_29")

  log_info("Fill strategy:")
  log_info("  - Unknown value for: card4, card6, P_emaildomain, ...")
  log_info("  - Mode for: id_15, id_16, id_28, id_29")
  log_info("  - Median for numeric columns")
  log_info("  - Mode for other columns")

  for (col_name in names(data)) {
    x <- data[[col_name]]
    if (any(is.na(x))) {
      na_count <- sum(is.na(x))
      if (col_name %in% fill_unknown_cols) {
        data[[col_name]][is.na(x)] <- "unknown"
      } else if (col_name %in% fill_mode_cols) {
        data[[col_name]][is.na(x)] <- getmode(x)
      } else if (is.numeric(x)) {
        data[[col_name]][is.na(x)] <- median(x, na.rm = TRUE)
      } else {
        data[[col_name]][is.na(x)] <- getmode(x)
      }
      log_info(sprintf("Filled %d missing values in %s", na_count, col_name)) # nolint: line_length_linter.
    }
  }

  # Verification
  final_missing <- colSums(is.na(data))
  if (any(final_missing > 0)) {
    log_warn("Some columns still have missing values:")
    for (col in names(final_missing)[final_missing > 0]) {
      log_warn(sprintf("  - %s: %d missing", col, final_missing[col]))
    }
  } else {
    log_info("All missing values successfully filled")
  }

  return(data)
}

# Analyze and visualize missing values
missing_values_analysis <- function(missing_info, visualize = TRUE) {
  if (any(missing_info$Missing_Count > 0)) {
    log_info("Missing Values Analysis")

    # Print missing values summary
    log_info(sprintf("• Total missing values: %d", sum(missing_info$Missing_Count))) # nolint: line_length_linter.
    log_info(sprintf("• Affected columns: %d", nrow(missing_info)))

    # Print details for each column with missing values
    log_info("• Details by column:")
    for (i in 1:nrow(missing_info)) {
      log_info(sprintf("  - %s: %d (%.1f%%)", missing_info$Feature[i], missing_info$Missing_Count[i], missing_info$Missing_Percent[i])) # nolint: line_length_linter.
    }

    if (visualize) {
      log_info("• Generating visualizations")

      # Missing Values Count Bar Chart
      plot_count <- ggplot(
        missing_info,
        aes(x = reorder(Feature, -Missing_Count), y = Missing_Count)
      ) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(
          title = "Missing Values Count",
          x = "Feature",
          y = "Missing Values Count"
        ) +
        theme_minimal()

      # Missing Values Percentage Bar Chart
      plot_percent <- ggplot(
        missing_info,
        aes(x = reorder(Feature, -Missing_Percent), y = Missing_Percent)
      ) +
        geom_bar(stat = "identity", fill = "lightcoral") +
        coord_flip() +
        labs(
          title = "Missing Values Percentage",
          x = "Feature",
          y = "Missing Values Percentage (%)"
        ) +
        theme_minimal()

      # Save plots
      ggsave("./pics/missing_values_count.png", plot = plot_count, width = 8, height = 6) # nolint: line_length_linter.
      ggsave("./pics/missing_values_percentage.png", plot = plot_percent, width = 8, height = 6) # nolint: line_length_linter.

      log_info("• Saved visualizations to 'missing_values_count.png' and 'missing_values_percentage.png'") # nolint: line_length_linter.
    }
  } else {
    log_info("No missing values found")
  }
}

# Handle missing values
handle_missing_values <- function(data, remove_threshold = 90, visualize = TRUE) { # nolint: line_length_linter.
  log_info("Handling Missing Values")
  log_info(sprintf("• Removal threshold: %d%%", remove_threshold))

  # Convert empty strings to NA
  log_info("• Converting empty strings to NA")
  data <- convert_empty_strings_to_na(data)

  # Analyze missing values
  missing_values <- colSums(is.na(data))
  missing_percentage <- (missing_values / nrow(data)) * 100

  missing_info <- data.frame(
    Feature = names(missing_values),
    Missing_Count = missing_values,
    Missing_Percent = round(missing_percentage, 2)
  ) %>%
    filter(Missing_Count > 0) %>%
    arrange(desc(Missing_Percent))

  # Analyze and visualize missing values
  missing_values_analysis(missing_info, visualize)

  # Remove high missing columns
  data <- remove_high_missing_columns(data, missing_percentage, remove_threshold) # nolint: line_length_linter.

  # Fill remaining missing values
  log_info("• Filling remaining missing values")
  data <- fill_missing_values(data)

  return(data)
}

# 2.3 Univariate Analysis
univariate_analysis <- function(data, output_dir = "pics/univariate") {
  log_info("Starting Univariate Analysis")

  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    log_info(sprintf("Output directory created: %s", output_dir))
  } else {
    log_info(sprintf("Output directory already exists: %s", output_dir))
  }

  # Identify numeric and factor variables
  numeric_vars <- names(data)[sapply(data, is.numeric)]
  factor_vars <- names(data)[sapply(data, function(x) is.factor(x) | is.character(x))] # nolint: line_length_linter.
  log_info(sprintf("Identified %d numeric variables and %d factor variables", length(numeric_vars), length(factor_vars))) # nolint: line_length_linter.

  # Plot numeric variables
  walk(numeric_vars, ~ {
    var_name <- .x
    log_info(sprintf("Plotting histogram and boxplot for numeric variable: %s", var_name)) # nolint: line_length_linter.

    create_and_save_plot(
      data = data,
      aes_mapping = aes(x = .data[[var_name]]),
      geom_func = geom_histogram,
      geom_args = list(bins = 30, fill = "lightgreen", color = "black"),
      labs_args = list(title = paste(var_name, "Distribution"), x = var_name, y = "Frequency"), # nolint: line_length_linter.
      theme_args = list(),
      output_dir = output_dir,
      filename = paste0(var_name, "_hist")
    )

    create_and_save_plot(
      data = data,
      aes_mapping = aes(y = .data[[var_name]]),
      geom_func = geom_boxplot,
      geom_args = list(fill = "lightblue"),
      labs_args = list(title = paste(var_name, "Boxplot"), y = var_name),
      theme_args = list(),
      output_dir = output_dir,
      filename = paste0(var_name, "_box")
    )
  })

  # Plot factor variables
  walk(factor_vars, ~ {
    var_name <- .x
    log_info(sprintf("Plotting bar chart for factor variable: %s", var_name))

    create_and_save_plot(
      data = data,
      aes_mapping = aes(x = .data[[var_name]]),
      geom_func = geom_bar,
      geom_args = list(fill = "lightcoral"),
      labs_args = list(title = paste(var_name, "Bar Chart"), x = var_name, y = "Count"), # nolint: line_length_linter.
      theme_args = list(axis.text.x = element_text(angle = 90, hjust = 1)),
      output_dir = output_dir,
      filename = paste0(var_name, "_bar")
    )
  })

  # Save frequency tables
  freq_file <- file.path(output_dir, "frequency_tables.txt")
  log_info(sprintf("Saving frequency tables for factor variables to: %s", freq_file)) # nolint: line_length_linter.
  sink(freq_file)
  walk(factor_vars, ~ {
    var_name <- .x
    cat(sprintf("\n%s Frequency Table:\n", var_name))
    print(table(data[[var_name]]))
  })
  sink()

  log_info("Univariate Analysis completed")
}

# 2.4 Bivariate Analysis
bivariate_analysis <- function(data, output_dir = "pics/bivariate") {
  log_info("Starting Bivariate Analysis")

  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    log_info(sprintf("Output directory created: %s", output_dir))
  } else {
    log_info(sprintf("Output directory already exists: %s", output_dir))
  }

  # Identify numeric and factor variables excluding isFraud
  numeric_vars <- setdiff(names(data)[sapply(data, is.numeric)], "isFraud")
  factor_vars <- setdiff(names(data)[sapply(data, function(x) is.factor(x) | is.character(x))], "isFraud") # nolint: line_length_linter.
  log_info(sprintf("Identified %d numeric variables and %d factor variables for bivariate analysis", length(numeric_vars), length(factor_vars))) # nolint: line_length_linter.

  # Relationship between numeric variables and isFraud
  walk(numeric_vars, ~ {
    var_name <- .x
    log_info(sprintf("Analyzing relationship between numeric variable: %s and target variable isFraud", var_name)) # nolint: line_length_linter.

    create_and_save_plot(
      data = data,
      aes_mapping = aes(x = .data[[var_name]], fill = as.factor(isFraud)),
      geom_func = geom_histogram,
      geom_args = list(bins = 30, position = "dodge"),
      labs_args = list(title = paste(var_name, "Relationship with Fraud"), x = var_name, y = "Frequency", fill = "Fraud"), # nolint: line_length_linter.
      theme_args = list(),
      output_dir = output_dir,
      filename = paste0(var_name, "_fraud")
    )

    create_and_save_plot(
      data = data,
      aes_mapping = aes(x = as.factor(isFraud), y = .data[[var_name]]),
      geom_func = geom_boxplot,
      geom_args = list(fill = "lightblue"),
      labs_args = list(title = paste(var_name, "Distribution by Fraud Status"), x = "Fraud", y = var_name), # nolint: line_length_linter.
      theme_args = list(),
      output_dir = output_dir,
      filename = paste0(var_name, "_fraud_box")
    )
  })

  # Relationship between factor variables and isFraud
  walk(factor_vars, ~ {
    var_name <- .x
    log_info(sprintf("Analyzing relationship between factor variable: %s and target variable isFraud", var_name)) # nolint: line_length_linter.

    create_and_save_plot(
      data = data,
      aes_mapping = aes(x = .data[[var_name]], fill = as.factor(isFraud)),
      geom_func = geom_bar,
      geom_args = list(position = "dodge"),
      labs_args = list(title = paste(var_name, "Relationship with Fraud"), x = var_name, y = "Count", fill = "Fraud"), # nolint: line_length_linter.
      theme_args = list(axis.text.x = element_text(angle = 90, hjust = 1)),
      output_dir = output_dir,
      filename = paste0(var_name, "_fraud")
    )
  })

  log_info("Bivariate Analysis completed")
}

# 2.5 Correlation Analysis
correlation_analysis <- function(data) {
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  numeric_cols <- setdiff(numeric_cols, c("isFraud", "TransactionID"))

  # Calculate standard deviation
  sds <- sapply(data[, numeric_cols, drop = FALSE], sd, na.rm = TRUE)
  numeric_cols <- numeric_cols[sds > 0]

  if (length(numeric_cols) > 1) {
    log_info("Correlation Analysis")
    correlation_matrix <- cor(data[numeric_cols])
    high_corr_pairs <- which(
      upper.tri(correlation_matrix) & abs(correlation_matrix) > 0.7,
      arr.ind = TRUE
    )

    if (nrow(high_corr_pairs) > 0) {
      log_info("High correlation feature pairs (|r| > 0.7):")
      # Print each high correlation pair
      for (i in 1:nrow(high_corr_pairs)) {
        feature1 <- numeric_cols[high_corr_pairs[i, 1]]
        feature2 <- numeric_cols[high_corr_pairs[i, 2]]
        corr_value <- correlation_matrix[high_corr_pairs[i, 1], high_corr_pairs[i, 2]] # nolint: line_length_linter.

        log_info(sprintf(
          "• %s - %s: %.3f",
          feature1,
          feature2,
          corr_value
        ))
      }
      # Print the total number of high correlation pairs
      log_info(sprintf("\nTotal high correlation pairs: %d", nrow(high_corr_pairs))) # nolint: line_length_linter.
    } else {
      log_info("No high correlation feature pairs (|r| > 0.7)")
    }
    # Save correlation heatmap
    png("./pics/correlation_heatmap.png", width = 800, height = 600)
    corrplot(correlation_matrix, method = "color", tl.cex = 0.8)
    dev.off()
    log_info("Correlation heatmap saved as 'correlation_heatmap.png'")
  } else {
    log_warn("Not enough numeric columns for correlation analysis.")
  }
}

# 2.6 Target Variable Analysis
target_variable_analysis <- function(data) {
  if ("isFraud" %in% names(data)) {
    log_info("Target Variable Analysis")
    fraud_ratio <- mean(data$isFraud, na.rm = TRUE)
    log_info(sprintf("Fraud Ratio: %.2f%%", fraud_ratio * 100))
    log_info(sprintf("Positive-Negative Sample Ratio: 1:%.2f", (1 - fraud_ratio) / fraud_ratio)) # nolint: line_length_linter.
  }
}

# 2.7 EDA Main Function
perform_eda <- function(data, remove_threshold = 90, visualize = TRUE) { # nolint: line_length_linter.
  log_info("Starting Exploratory Data Analysis (EDA)")

  # 1. Print basic information
  basic_information(data)

  # 2. Analyze and handle missing values
  data <- handle_missing_values(data, remove_threshold, visualize)

  # 3. Univariate Analysis
  if (visualize) {
    univariate_analysis(data)
  }

  # 4. Bivariate Analysis
  if (visualize) {
    bivariate_analysis(data)
  }

  # 5. Correlation Analysis
  correlation_analysis(data)

  # 6. Target Variable Analysis
  target_variable_analysis(data)

  log_info("EDA completed.")

  return(data)
}

# ============================================
# 3. Data preprocessing
# ============================================

# 3.1 Handle outliers
handle_outliers <- function(data, method = "iqr", threshold = 1.5, exclude_columns = c("TransactionID", "isFraud")) { # nolint: line_length_linter.
  log_info("Handling Outliers")
  log_info(sprintf("• Method: %s", method))
  log_info(sprintf("• Threshold: %.2f", threshold))

  # Identify numeric columns
  numeric_cols <- sapply(data, is.numeric)

  # Separate fraud and non-fraud data
  fraud_data <- data[data$isFraud == 1, ]
  non_fraud_data <- data[data$isFraud == 0, ]

  # Process numeric columns
  for (col in names(data)[numeric_cols]) {
    if (col %in% exclude_columns) {
      log_info(sprintf("  - Skipping: %s", col))
      next
    }

    # Non-fraud data
    x <- non_fraud_data[[col]]
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1

    lower_bound <- Q1 - threshold * IQR_val
    upper_bound <- Q3 + threshold * IQR_val

    outliers <- x < lower_bound | x > upper_bound

    if (any(outliers, na.rm = TRUE)) {
      log_info(sprintf("  - %s: found %d outliers", col, sum(outliers, na.rm = TRUE))) # nolint: line_length_linter.

      if (method == "iqr") {
        # Handle outliers for non-fraud samples
        non_fraud_data[[col]][x < lower_bound] <- Q1
        non_fraud_data[[col]][x > upper_bound] <- Q3
        log_info(sprintf("    • Applied IQR method (Q1: %.2f, Q3: %.2f)", Q1, Q3)) # nolint: line_length_linter.
      } else if (method == "winsorize") {
        # Process outliers in non-fraud samples using the Winsorize method
        non_fraud_data[[col]] <- Winsorize(x, probs = c(0.05, 0.95))
        log_info("    • Applied Winsorize method")
      } else {
        log_warn("Unknown processing method, no outlier processing applied.") # nolint: line_length_linter.
      }
    } else {
      log_info(sprintf("  - %s: no outliers found", col))
    }
  }

  # Combine processed non-fraud and fraud data
  processed_data <- rbind(non_fraud_data, fraud_data)
  # Maintain the original data frame's row order
  processed_data <- processed_data[order(processed_data$TransactionID), ]

  log_info("Outliers handling completed")
  return(processed_data)
}

# 3.2 Feature Engineering
feature_engineering <- function(data) {
  log_info("Starting Feature Engineering")

  # Time Features
  if ("TransactionDT" %in% names(data)) {
    log_info("• Processing time features")
    data$TransactionHour <- (data$TransactionDT %% 86400) %/% 3600
    data$TransactionDayOfWeek <- (data$TransactionDT %/% 86400) %% 7 + 1
    data$TransactionDayOfMonth <- (data$TransactionDT %/% 86400) %% 30 + 1
    log_info("  - Created: TransactionHour, TransactionDayOfWeek, TransactionDayOfMonth") # nolint: line_length_linter.
  } else {
    log_warn("TransactionDT not found, skipping time features")
  }

  # Amount Features
  if ("TransactionAmt" %in% names(data)) {
    log_info("• Processing amount features")

    # Logarithmic transformation
    data$TransactionAmt_Log <- log1p(data$TransactionAmt)
    log_info("  - Created: TransactionAmt_Log")

    # Binning
    breaks <- unique(quantile(data$TransactionAmt, probs = seq(0, 1, 0.1), na.rm = TRUE)) # nolint: line_length_linter.
    # If all quantiles are the same, use equal width binning
    if (length(breaks) < 2) {
      breaks <- seq(min(data$TransactionAmt), max(data$TransactionAmt), length.out = 11) # nolint: line_length_linter.
      log_info("  - Using equal width binning for TransactionAmt")
    } else {
      log_info("  - Using quantile-based binning for TransactionAmt")
    }

    data$TransactionAmt_Bin <- cut(
      data$TransactionAmt,
      breaks = breaks,
      labels = 1:(length(breaks) - 1),
      include.lowest = TRUE
    )
    log_info(sprintf("  - Created TransactionAmt_Bin with %d bins", length(breaks) - 1)) # nolint: line_length_linter.
  } else {
    log_warn("TransactionAmt not found, skipping amount features")
  }

  # Special handling for id_14
  if ("id_14" %in% names(data)) {
    log_info("• Processing id_14 feature")
    data$id_14 <- (data$id_14 / 60 + 10) / 20
    log_info("  - Normalized id_14 to 0-1 range")
  } else {
    log_warn("id_14 not found, skipping processing")
  }

  # Convert logical/boolean columns to 0/1
  logical_cols <- names(data)[sapply(data, is.logical)]
  if (length(logical_cols) > 0) {
    log_info("• Converting logical columns to numeric")
    for (col in logical_cols) {
      data[[col]] <- as.numeric(data[[col]])
    }
    log_info(sprintf("  - Converted %d logical columns", length(logical_cols))) # nolint: line_length_linter.
  } else {
    log_info("• No logical columns to convert")
  }

  log_info("Feature Engineering completed")
  return(data)
}

# 3.3 Encode Categorical Variables
# Helper Function: Target Encoding
target_encode <- function(data, col, use_smoothing = TRUE) {
  if (use_smoothing) {
    encoding <- data %>%
      group_by(!!sym(col)) %>%
      summarize(encoded = mean(isFraud, na.rm = TRUE), count = n()) %>%
      mutate(
        global_mean = mean(data$isFraud, na.rm = TRUE),
        lambda = 1 / (1 + exp(-count / 10)),
        encoded = lambda * encoded + (1 - lambda) * global_mean
      )
    return(encoding$encoded[match(data[[col]], encoding[[col]])])
  } else {
    encoding <- aggregate(data$isFraud ~ data[[col]], FUN = mean)
    return(encoding$`data$isFraud`[match(data[[col]], encoding$`data[[col]]`)])
  }
}

# Helper Function: One-Hot Encoding
one_hot_encode <- function(data, col) {
  dummy_vars <- model.matrix(~ data[[col]] - 1)
  colnames(dummy_vars) <- paste0(col, "_", levels(factor(data[[col]])))
  data <- cbind(data, dummy_vars)
  data[[col]] <- NULL
  return(data)
}

# Encode Categorical Variables
encode_categorical <- function(data) {
  log_info("Encoding Categorical Variables")

  # Get all categorical columns
  categorical_cols <- names(data)[sapply(data, function(x) is.character(x) | is.factor(x))] # nolint: line_length_linter.
  if (length(categorical_cols) == 0) {
    log_info("• No categorical columns found")
    return(data)
  }

  log_info(sprintf("• Found %d categorical columns", length(categorical_cols))) # nolint: line_length_linter.

  # Predefined Variable Types
  predefined_cols <- list(
    high_cardinality = c("id_31", "id_33", "DeviceInfo"),
    business_meaningful = c("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "id_30", "id_34", "DeviceType"), # nolint: line_length_linter.
    binary = c("id_12", "id_16", "id_28", "id_29"),
    low_cardinality = c("M4", "id_15")
  )

  # Log predefined categories
  log_info("• Predefined categories:")
  for (category in names(predefined_cols)) {
    cols <- predefined_cols[[category]]
    if (length(cols) > 0) {
      log_info(sprintf("  - %s: %s", category, paste(cols, collapse = ", "))) # nolint: line_length_linter.
    }
  }

  # Process all categorical variables
  log_info("• Processing columns:")
  for (col in categorical_cols) {
    n_categories <- length(unique(data[[col]]))
    log_info(sprintf("  - %s (%d unique values)", col, n_categories)) # nolint: line_length_linter.

    # Determine encoding method
    if (col %in% predefined_cols$high_cardinality || n_categories > 100) {
      log_info("    • Using target encoding (smoothed)")
      data[[col]] <- target_encode(data, col, use_smoothing = TRUE)
    } else if (col %in% predefined_cols$business_meaningful || (n_categories > 10 && n_categories <= 100)) { # nolint: line_length_linter.
      log_info("    • Using target encoding")
      data[[col]] <- target_encode(data, col, use_smoothing = FALSE)
    } else if (col %in% predefined_cols$binary || n_categories == 2) {
      log_info("    • Using binary encoding")
      data[[col]] <- as.numeric(factor(data[[col]])) - 1
    } else if (col %in% predefined_cols$low_cardinality || n_categories <= 10) {
      log_info("    • Using one-hot encoding")
      data <- one_hot_encode(data, col)
      log_info(sprintf("      - Created %d new columns", n_categories)) # nolint: line_length_linter.
    }
  }

  # Final summary
  numeric_cols <- sum(sapply(data, is.numeric))
  factor_cols <- sum(sapply(data, is.factor))
  log_info(sprintf("• Final column composition: Numeric: %d, Factor: %d", numeric_cols, factor_cols)) # nolint: line_length_linter.
  log_info("Categorical encoding completed")

  return(data)
}

# 3.4 Feature Selection
feature_selection <- function(data, correlation_threshold = 0.8) {
  log_info("Feature Selection")
  log_info(sprintf("• Correlation threshold: %.2f", correlation_threshold))

  # Save original column names
  original_cols <- names(data)

  # Identify one-hot encoded groups
  onehot_groups <- list(
    TransactionAmt_Bin = grep("^TransactionAmt_Bin_", names(data), value = TRUE), # nolint: line_length_linter.
    M4 = grep("^M4_", names(data), value = TRUE),
    id_15 = grep("^id_15_", names(data), value = TRUE)
  )

  # Log one-hot encoded groups
  log_info("• One-hot encoded groups:")
  for (group_name in names(onehot_groups)) {
    log_info(sprintf("  - %s: %s", group_name, paste(onehot_groups[[group_name]], collapse = ", "))) # nolint: line_length_linter.
  }

  # Protected columns
  fully_protected_cols <- c(
    "isFraud",
    unlist(onehot_groups),
    "TransactionHour", "TransactionDayOfWeek", "TransactionDayOfMonth"
  )
  encoded_categorical_cols <- c(
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "id_12", "id_16", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34"
  )

  log_info("• Protected columns:")
  log_info(sprintf("  - Fully protected: %s", paste(fully_protected_cols, collapse = ", "))) # nolint: line_length_linter.
  log_info(sprintf("  - Encoded categorical: %s", paste(encoded_categorical_cols, collapse = ", "))) # nolint: line_length_linter.

  # Zero-variance features
  log_info("• Checking zero-variance features")
  analysis_cols <- setdiff(names(data), fully_protected_cols)
  if (length(analysis_cols) > 0) {
    zero_var_cols <- nearZeroVar(data[, analysis_cols, drop = FALSE], saveMetrics = TRUE) # nolint: line_length_linter.
    zero_var_features <- rownames(zero_var_cols)[zero_var_cols$nzv]
    if (length(zero_var_features) > 0) {
      data <- data[, !names(data) %in% zero_var_features]
      log_info(sprintf("  - Removed zero-variance features: %s", paste(zero_var_features, collapse = ", "))) # nolint: line_length_linter.
    } else {
      log_info("  - No zero-variance features found")
    }
  }

  # Correlation analysis
  log_info("• Performing correlation analysis")
  correlation_cols <- setdiff(names(data)[sapply(data, is.numeric)], fully_protected_cols) # nolint: line_length_linter.

  if (length(correlation_cols) > 1) {
    cor_matrix <- cor(data[, correlation_cols, drop = FALSE], use = "pairwise.complete.obs") # nolint: line_length_linter.

    # Check encoded categorical variables
    for (col in intersect(encoded_categorical_cols, correlation_cols)) {
      high_cor_with_col <- names(which(abs(cor_matrix[col, ]) > correlation_threshold)) # nolint: line_length_linter.
      high_cor_with_col <- setdiff(high_cor_with_col, col)
      if (length(high_cor_with_col) > 0) {
        log_info(sprintf("  - %s correlated with: %s", col, paste(high_cor_with_col, collapse = ", "))) # nolint: line_length_linter.
      }
    }

    # Remove highly correlated features
    high_cor <- findCorrelation(cor_matrix, cutoff = correlation_threshold)
    if (length(high_cor) > 0) {
      cor_features <- correlation_cols[high_cor]
      encoded_to_remove <- intersect(cor_features, encoded_categorical_cols)

      if (length(encoded_to_remove) > 0) {
        log_warn(sprintf("  - Removing encoded categorical variables due to high correlation: %s", paste(encoded_to_remove, collapse = ", "))) # nolint: line_length_linter.
      }

      data <- data[, !names(data) %in% cor_features]
      log_info(sprintf("• Removed correlated features: %s", paste(cor_features, collapse = ", "))) # nolint: line_length_linter.
    } else {
      log_info("• No high correlations found")
    }
  }

  # Results summary
  removed_features <- setdiff(original_cols, names(data))
  log_info("Feature Selection Summary:")
  log_info(sprintf("  - Features before: %d", length(original_cols)))
  log_info(sprintf("  - Features after: %d", ncol(data)))
  log_info(sprintf("  - Features removed: %d", length(removed_features))) # nolint: line_length_linter.

  log_info("Feature Selection completed")
  return(data)
}

# 3.5 Scale Features
scale_features <- function(data, method = c("standardization", "minmax", "robust")) { # nolint: line_length_linter.
  method <- match.arg(method)
  log_info("Feature Scaling")
  log_info(sprintf("• Method: %s", method))

  # Binary check function
  binary_check <- function(x) {
    if (is.numeric(x)) {
      unique_vals <- unique(na.omit(x))
      return(length(unique_vals) == 2 && all(unique_vals %in% c(0, 1)))
    }
    return(FALSE)
  }

  # Exclude columns
  exclude_cols <- c(
    "isFraud",
    grep("TransactionAmt_Bin_|^M4_|^id_15_", names(data), value = TRUE),
    c("TransactionHour", "TransactionDayOfWeek", "TransactionDayOfMonth"),
    c(
      "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
      "DeviceType", "DeviceInfo"
    ),
    "id_14",
    names(data)[sapply(data, binary_check)]
  )

  # Get numeric columns that need to be scaled
  cols_to_scale <- setdiff(names(data)[sapply(data, is.numeric)], exclude_cols)

  if (length(cols_to_scale) == 0) {
    log_info("• No features require scaling")
    return(data)
  }

  # Scale the features
  log_info(sprintf("• Scaling features: %s", paste(cols_to_scale, collapse = ", "))) # nolint: line_length_linter.

  for (col in cols_to_scale) {
    x <- data[[col]]
    if (var(x, na.rm = TRUE) == 0) next

    data[[col]] <- switch(method,
      "standardization" = (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE),
      "minmax" = (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)), # nolint: line_length_linter.
      "robust" = (x - median(x, na.rm = TRUE)) / IQR(x, na.rm = TRUE)
    )
  }

  log_info("Feature Scaling completed")
  return(data)
}

# 3.6 Handle Imbalance
handle_imbalance <- function(data, method = c("smote", "rose", "adasyn")) {
  method <- match.arg(method)
  log_info("Handling Class Imbalance")
  log_info(sprintf("• Method: %s", toupper(method)))

  # Record the original class distribution
  orig_dist <- table(data$isFraud)
  log_info(sprintf(
    "• Original distribution: Non-fraud: %d (%.1f%%), Fraud: %d (%.1f%%)",
    orig_dist[1], 100 * orig_dist[1] / sum(orig_dist),
    orig_dist[2], 100 * orig_dist[2] / sum(orig_dist)
  ))

  # Ensure the target variable is a factor
  data$isFraud <- as.factor(data$isFraud)

  # Sampling according to different methods
  log_info("• Applying balancing")
  balanced_data <- switch(method,
    "smote" = {
      log_info("  - Using SMOTE with K=5")
      smote_result <- smotefamily::SMOTE(
        X = data[, -which(names(data) == "isFraud")],
        target = data$isFraud,
        K = 5,
        dup_size = 3
      )
      smote_df <- as.data.frame(smote_result$data)
      names(smote_df)[names(smote_df) == "class"] <- "isFraud"
      smote_df
    },
    "rose" = {
      log_info("  - Using ROSE")
      ROSE(isFraud ~ ., data = data)$data
    },
    "adasyn" = {
      log_info("  - Using ADASYN with K=5")
      adasyn_result <- smotefamily::ADAS(
        X = data[, -which(names(data) == "isFraud")],
        target = data$isFraud,
        K = 5
      )
      adasyn_df <- as.data.frame(adasyn_result$data)
      names(adasyn_df)[names(adasyn_df) == "class"] <- "isFraud"
      adasyn_df
    }
  )

  # Ensure the target variable of the result is a factor
  balanced_data$isFraud <- as.factor(balanced_data$isFraud)

  # Output the final distribution
  final_dist <- table(balanced_data$isFraud)
  log_info(sprintf(
    "• Final distribution: Non-fraud: %d (%.1f%%), Fraud: %d (%.1f%%)",
    final_dist[1], 100 * final_dist[1] / sum(final_dist),
    final_dist[2], 100 * final_dist[2] / sum(final_dist)
  ))

  # Summary
  imbalance_ratio_before <- orig_dist[1] / orig_dist[2]
  imbalance_ratio_after <- final_dist[1] / final_dist[2]
  log_info(sprintf("• Imbalance ratio: %.2f:1 → %.2f:1", imbalance_ratio_before, imbalance_ratio_after)) # nolint: line_length_linter.

  log_info("Class balancing completed")
  return(balanced_data)
}

# 3.7 Data Preprocessing Main Function
preprocess_raw_data <- function(
    data,
    handle_outliers_flag = TRUE,
    feature_engineering_flag = TRUE,
    feature_selection_flag = TRUE,
    encode_categorical_flag = TRUE,
    scale_features_flag = TRUE) {
  log_info("Starting Data Preprocessing Pipeline")

  # Initial data summary
  log_info(sprintf("• Initial data summary: %d rows × %d columns", nrow(data), ncol(data))) # nolint: line_length_linter.

  # Initial class distribution
  init_dist <- table(data$isFraud)
  log_info(sprintf(
    "• Class distribution: Non-fraud: %d (%.1f%%), Fraud: %d (%.1f%%)",
    init_dist[1], 100 * init_dist[1] / sum(init_dist),
    init_dist[2], 100 * init_dist[2] / sum(init_dist)
  ))

  # Handle outliers
  if (handle_outliers_flag) {
    data <- handle_outliers(data)
  } else {
    log_info("• Outlier handling: SKIPPED")
  }

  # Feature Engineering
  if (feature_engineering_flag) {
    data <- feature_engineering(data)
  } else {
    log_info("• Feature engineering: SKIPPED")
  }

  # Encode categorical variables
  if (encode_categorical_flag) {
    data <- encode_categorical(data)
  } else {
    log_info("• Categorical encoding: SKIPPED")
  }

  # Feature Selection
  if (feature_selection_flag) {
    data <- feature_selection(data)
  } else {
    log_info("• Feature selection: SKIPPED")
  }

  # Scale features
  if (scale_features_flag) {
    data <- scale_features(data, method = "robust")
  } else {
    log_info("• Feature scaling: SKIPPED")
  }

  # Final Summary
  log_info("Data Preprocessing Pipeline completed")
  log_info(sprintf("• Final data summary: %d rows × %d columns", nrow(data), ncol(data))) # nolint: line_length_linter.

  return(data)
}

# ============================================
# 4. Model Training Functions
# ============================================

# 4.1 Logistic Regression
train_logistic_regression <- function(train_data) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  train_data$isFraud <- factor(
    train_data$isFraud,
    levels = c(0, 1),
    labels = c("NonFraud", "Fraud")
  )

  train_control <- trainControl(
    method = "cv", # Use cross-validation
    number = 5, # 5-fold cross-validation
    classProbs = TRUE, # Calculate class probabilities
    allowParallel = TRUE # Allow parallel processing
  )

  model <- train(
    isFraud ~ .,
    data = train_data,
    method = "glmnet",
    trControl = train_control
  )

  return(model)
}

# 4.2 Random Forest
train_random_forest <- function(train_data) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  # Ensure the target variable is a factor
  train_data$isFraud <- as.factor(train_data$isFraud)

  model <- randomForest(
    isFraud ~ .,
    data = train_data,
    ntree = 100,
    mtry = floor(sqrt(ncol(train_data) - 1)),
    importance = TRUE,
    nodesize = 5,
    type = "classification"
  )

  return(model)
}

train_random_forest_parallel <- function(train_data, num_cores = detectCores() - 1) { # nolint: line_length_linter.
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  # Ensure the target variable is a factor
  train_data$isFraud <- as.factor(train_data$isFraud)

  model <- ranger(
    formula = isFraud ~ .,
    data = train_data,
    num.trees = 100,
    mtry = floor(sqrt(ncol(train_data) - 1)),
    importance = "impurity",
    min.node.size = 5,
    num.threads = num_cores,
    classification = TRUE
  )

  return(model)
}

# 4.3 Gradient Boosting
train_gradient_boosting <- function(train_data, num_cores = detectCores() - 1) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  dtrain <- xgb.DMatrix(
    data = as.matrix(train_data[, -which(names(train_data) == "isFraud")]),
    label = as.numeric(train_data$isFraud)
  )

  params <- list(
    objective = "binary:logistic",
    eta = 0.05,
    max_depth = 8,
    min_child_weight = 1,
    subsample = 0.7,
    colsample_bytree = 0.7,
    nthread = num_cores
  )

  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 200,
    verbose = 0
  )

  return(model)
}

# 4.4 LightGBM
train_lightgbm <- function(train_data, num_cores = detectCores() - 1) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  dtrain <- lgb.Dataset(
    data = as.matrix(train_data[, -which(names(train_data) == "isFraud")]),
    label = as.numeric(train_data$isFraud)
  )

  params <- list(
    objective = "binary",
    metric = "auc",
    learning_rate = 0.05,
    num_leaves = 64,
    feature_fraction = 0.7,
    bagging_fraction = 0.7,
    bagging_freq = 5,
    verbosity = -1,
    nthread = num_cores
  )

  model <- lgb.train(
    params = params,
    data = dtrain,
    nrounds = 200,
    verbose = 0,
    early_stopping_rounds = 10,
    valid = list(train = dtrain)
  )

  return(model)
}

# 4.5 SVM
train_svm <- function(train_data, num_cores = detectCores() - 1) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  model <- svm(
    isFraud ~ .,
    data = train_data,
    kernel = "linear",
    probability = TRUE,
    scale = TRUE,
    cost = 1,
    gamma = 0.1,
    parallel = TRUE,
    cores = num_cores
  )

  return(model)
}

# 4.6 Neural Network
train_neural_network <- function(train_data) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  x_train <- as.matrix(train_data[, -which(names(train_data) == "isFraud")])
  y_train <- as.numeric(train_data$isFraud)
  model <- nnet(
    x_train,
    y_train,
    size = 10,
    maxit = 100,
    linout = FALSE,
    decay = 0.1,
    trace = FALSE
  )

  return(model)
}

# 4.7 Deep Neural Network
train_deep_neural_network <- function(train_data) {
  # Check if the label is 0 or 1
  if (!all(train_data$isFraud %in% c(0, 1))) {
    log_error("Label must be 0 or 1")
    stop("Label must be 0 or 1")
  }

  # Ensure the target variable is a factor
  train_data$isFraud <- as.factor(train_data$isFraud)

  model <- neuralnet(
    formula = isFraud ~ .,
    data = train_data,
    hidden = c(64, 32, 16), # Three hidden layers
    act.fct = "logistic", # Sigmoid activation function
    linear.output = FALSE, # Binary classification problem
    threshold = 0.1, # Convergence threshold
    stepmax = 1e6, # Maximum number of iterations
    algorithm = "rprop+" # Resilient backpropagation algorithm
  )

  return(model)
}

# 4.8 Model Training Main Function
train_model <- function(train_data, method) {
  log_info(sprintf("Training Model: %s", toupper(gsub("_", " ", method))))
  log_info(sprintf("• Training data size: %d samples", nrow(train_data)))

  model <- switch(method,
    "logistic_regression" = {
      log_info("• Using logistic regression with 5-fold CV")
      train_logistic_regression(train_data)
    },
    "random_forest" = {
      log_info("• Using random forest with 100 trees")
      train_random_forest(train_data)
    },
    "random_forest_parallel" = {
      cores <- detectCores() - 1
      log_info(sprintf("• Using parallel random forest with %d cores", cores))
      train_random_forest_parallel(train_data, cores)
    },
    "xgboost" = {
      cores <- detectCores() - 1
      log_info(sprintf("• Using XGBoost with %d cores", cores))
      train_gradient_boosting(train_data, cores)
    },
    "lightgbm" = {
      cores <- detectCores() - 1
      log_info(sprintf("• Using LightGBM with %d cores", cores))
      train_lightgbm(train_data, cores)
    },
    "svm" = {
      cores <- detectCores() - 1
      log_info(sprintf("• Using SVM with %d cores", cores))
      train_svm(train_data, cores)
    },
    "neural_network" = {
      log_info("• Using neural network with 10 hidden units")
      train_neural_network(train_data)
    },
    "deep_neural_network" = {
      log_info("• Using deep neural network with [64, 32, 16] architecture") # nolint: line_length_linter.
      train_deep_neural_network(train_data)
    },
    {
      log_error("Unsupported model method: %s", method)
      stop("Unsupported model method.")
    }
  )

  log_info("Model training completed")
  return(model)
}

# ============================================
# 5. Model Evaluation Functions
# ============================================

# 5.1 Evaluate Models
evaluate_models <- function(models, test_data) {
  log_info(sprintf("Evaluating models on test data: %d samples", nrow(test_data))) # nolint: line_length_linter.

  results <- list()
  x_test <- as.matrix(test_data[, -which(names(test_data) == "isFraud")])
  y_test <- as.numeric(test_data$isFraud)

  for (model_name in names(models)) {
    log_info(sprintf("Evaluating %s model", toupper(gsub("_", " ", model_name)))) # nolint: line_length_linter.
    model <- models[[model_name]]
    predictions <- predict_model(model, model_name, x_test)

    # Check if predictions are valid and calculate metrics
    if (is.null(predictions)) {
      log_warn("%s model did not produce predictions", model_name)
      next
    }

    metrics <- calculate_metrics(predictions, y_test)
    results[[model_name]] <- metrics
    print_results(model_name, metrics)

    # Save ROC curve
    roc_plot <- ggroc(metrics$roc)
    ggsave(paste0("./pics/", model_name, "_roc_curve.png"), plot = roc_plot, width = 8, height = 6) # nolint: line_length_linter.
    log_info(sprintf("ROC curve saved as '%s_roc_curve.png'", model_name))
  }

  return(results)
}

# 5.2 Predict Model
predict_model <- function(model, model_name, x_test) {
  switch(model_name,
    "logistic_regression" = {
      pred <- predict(model, newdata = as.data.frame(x_test), type = "prob")[, "Fraud"] # nolint: line_length_linter.
      return(pred)
    },
    "random_forest" = {
      pred <- predict(model, newdata = as.data.frame(x_test), type = "prob")[, 2] # nolint: line_length_linter.
      return(pred)
    },
    "random_forest_parallel" = {
      pred <- predict(model, data = as.data.frame(x_test), type = "response")
      return(as.numeric(pred$predictions) - 1)
    },
    "xgboost" = {
      pred <- predict(model, newdata = as.matrix(x_test))
      return(pred)
    },
    "lightgbm" = {
      pred <- predict(model, newdata = as.matrix(x_test))
      return(pred)
    },
    "svm" = {
      pred <- attr(predict(model, newdata = as.data.frame(x_test), probability = TRUE), "probabilities")[, 2] # nolint: line_length_linter.
      return(pred)
    },
    "neural_network" = {
      pred <- predict(model, x_test)
      return(as.vector(pred))
    },
    "deep_neural_network" = {
      pred_prob <- predict(model, newdata = as.data.frame(x_test))
      return(as.vector(pred_prob[, 2]))
    },
    {
      log_warn("Unsupported model type: %s", model_name)
      return(NULL)
    }
  )
}

# 5.3 Calculate Metrics
calculate_metrics <- function(y_pred, y_true) {
  # Check if the length of prediction and true values match
  if (length(y_pred) != length(y_true)) {
    log_error("Prediction and true values do not match in length.")
    stop("Prediction and true values do not match in length.")
  }

  # Binary Prediction
  y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)

  # Confusion Matrix
  conf_matrix <- table(
    Actual = factor(y_true, levels = c(0, 1)),
    Predicted = factor(y_pred_binary, levels = c(0, 1))
  )

  # True Positive, True Negative, False Positive, False Negative
  TP <- conf_matrix[2, 2]
  TN <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  FN <- conf_matrix[2, 1]

  # Accuracy, Precision, Recall, F1, AUC
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- if (TP + FP == 0) 0 else TP / (TP + FP)
  recall <- if (TP + FN == 0) 0 else TP / (TP + FN)
  f1 <- if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall) # nolint: line_length_linter.

  # AUC and ROC
  roc_obj <- roc(y_true, y_pred, quiet = TRUE)
  auc_val <- auc(roc_obj)

  # Return all metrics
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    auc = auc_val,
    roc = roc_obj,
    confusion_matrix = conf_matrix
  ))
}

# 5.4 Print Results
print_results <- function(model_name, results) {
  log_info("=======================================")
  log_info(sprintf("Results for %s Model: ", toupper(gsub("_", " ", model_name)))) # nolint: line_length_linter.
  log_info("=======================================")
  log_info("Performance Metrics:")
  log_info(sprintf("• Accuracy:  %.4f", results$accuracy))
  log_info(sprintf("• Precision: %.4f", results$precision))
  log_info(sprintf("• Recall:    %.4f", results$recall))
  log_info(sprintf("• F1-Score:  %.4f", results$f1))
  log_info(sprintf("• AUC:       %.4f", results$auc))
  log_info("Confusion Matrix:")
  conf_matrix <- results$confusion_matrix
  log_info(sprintf("              Predicted 0  Predicted 1"))
  log_info(sprintf("Actual 0      %-11d %-11d", conf_matrix[1, 1], conf_matrix[1, 2])) # nolint: line_length_linter.
  log_info(sprintf("Actual 1      %-11d %-11d", conf_matrix[2, 1], conf_matrix[2, 2])) # nolint: line_length_linter.
  log_info("=======================================\n")
}

# ============================================
# 6. Main Execution Function
# ============================================
run_fraud_detection <- function(
    data_path,
    target_variable = "isFraud",
    balance_methods = c("smote", "rose", "adasyn"),
    model_configs = list(),
    visualize = TRUE) {
  # Fraud Detection Start
  log_header("Fraud Detection Pipeline START")

  # Set random seed
  set.seed(42)
  log_info("Random seed set to 42")

  # 1. Load data
  log_header("STEP 1: LOADING DATA")
  log_info(sprintf("• Loading data from: %s", data_path))
  data <- read.csv(data_path)
  log_info(sprintf("• Loaded data: %d rows × %d columns", nrow(data), ncol(data))) # nolint: line_length_linter.

  # 2. Perform EDA
  log_header("STEP 2: EXPLORATORY DATA ANALYSIS")
  data <- perform_eda(data, remove_threshold = 90, visualize = visualize) # nolint: line_length_linter.

  # 3. Data preprocessing
  log_header("STEP 3: DATA PREPROCESSING")
  processed_data <- preprocess_raw_data(
    data,
    handle_outliers_flag = TRUE,
    feature_engineering_flag = TRUE,
    feature_selection_flag = TRUE,
    encode_categorical_flag = TRUE,
    scale_features_flag = TRUE
  )

  # 4. Split data
  log_header("STEP 4: DATA SPLITTING")
  train_index <- createDataPartition(processed_data[[target_variable]], p = 0.8, list = FALSE) # nolint: line_length_linter.
  train_data <- processed_data[train_index, ]
  test_data <- processed_data[-train_index, ]
  log_info(sprintf("• Training set: %d samples", nrow(train_data)))
  log_info(sprintf("• Testing set: %d samples", nrow(test_data)))

  # 5. Handling imbalanced data
  log_header("STEP 5: CLASS IMBALANCE HANDLING")
  log_info(sprintf("• Applying balancing methods: %s", paste(balance_methods, collapse = ", "))) # nolint: line_length_linter.
  balanced_train_data <- list()
  if (!is.null(balance_methods)) {
    for (method in balance_methods) {
      log_info(sprintf("• Applying %s method", toupper(method)))
      balanced_data <- handle_imbalance(train_data, method = method)
      balanced_train_data[[method]] <- balanced_data
    }
  }

  # 6. Model Training and Evaluation
  log_header("STEP 6: MODEL TRAINING & EVALUATION")
  results <- list()
  models_list <- list()

  for (model_config in model_configs) {
    model_name <- model_config$name
    log_info(sprintf("Model: %s", toupper(gsub("_", " ", model_name))))

    # Select training data
    if (isTRUE(model_config$need_balance)) {
      balance_method <- model_config$balance_method
      if (!is.null(balanced_train_data[[balance_method]])) {
        current_train_data <- balanced_train_data[[balance_method]]
        log_info(sprintf("• Using balanced data (%s)", balance_method))
      } else {
        current_train_data <- train_data
        log_warn(sprintf("• %s balanced data not found, using original data", balance_method)) # nolint: line_length_linter.
      }
    } else {
      current_train_data <- train_data
      log_info("• Using original training data")
    }

    # Training
    log_info("• Training Phase")
    start_time <- Sys.time()
    model <- train_model(current_train_data, model_name)
    models_list[[model_name]] <- model

    # Evaluating
    log_info("• Evaluation Phase")
    metrics <- evaluate_models(models_list[model_name], test_data)
    end_time <- Sys.time()

    # Record training completion and time information
    training_time <- difftime(end_time, start_time, units = "mins")
    log_info(sprintf("• Model completed in %.2f minutes", as.numeric(training_time))) # nolint: line_length_linter.
    log_info("=======================================\n")

    results[[model_name]] <- list(
      model = model,
      metrics = metrics[[model_name]]
    )
  }

  log_info("=== Fraud Detection Pipeline COMPLETE ===")

  return(results)
}

# ============================================
# 7. Execute the entire process
# ============================================

# Define Model Configs
default_model_configs <- list(
  list(name = "logistic_regression", need_balance = TRUE, balance_method = "smote"), # nolint: line_length_linter.
  list(name = "random_forest", need_balance = FALSE),
  list(name = "random_forest_parallel", need_balance = FALSE),
  list(name = "xgboost", need_balance = FALSE),
  list(name = "lightgbm", need_balance = FALSE),
  list(name = "svm", need_balance = TRUE, balance_method = "smote"),
  list(name = "neural_network", need_balance = TRUE, balance_method = "smote"),
  list(name = "deep_neural_network", need_balance = TRUE, balance_method = "smote") # nolint: line_length_linter.
)

model_configs <- list(
  list(name = "logistic_regression", need_balance = TRUE, balance_method = "smote"), # nolint: line_length_linter.
  list(name = "random_forest_parallel", need_balance = FALSE),
  list(name = "xgboost", need_balance = FALSE),
  list(name = "lightgbm", need_balance = FALSE),
  list(name = "neural_network", need_balance = TRUE, balance_method = "smote"),
  list(name = "deep_neural_network", need_balance = TRUE, balance_method = "smote") # nolint: line_length_linter.
)

# Execute the entire process
results <- run_fraud_detection(
  data_path = "./datasets/A1_data.csv",
  target_variable = "isFraud",
  balance_methods = c("smote", "rose", "adasyn"),
  model_configs = model_configs,
  visualize = FALSE
)

###################################################
# Close parallel cluster at the end of the script #
###################################################
if (exists("cl")) {
  stopCluster(cl)
  registerDoSEQ()
  log_info("Parallel cluster stopped at script end")
}
