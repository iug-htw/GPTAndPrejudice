
library(tidyverse)
library(readxl)


concepts <- list(
  marriage = c("marriage", "married", "marry", "wedding", "bride", "groom", 
               "spouse", "wife", "husband", "matrimony", "betrothal", "betrothed",
               "engagement", "engaged", "nuptial", "union", "match", "suitor",
               "courtship", "proposal", "wedlock", "conjugal", "marital"),
  
  family = c("family", "children", "child", "kids", "mother", "father", 
             "parent", "baby", "pregnancy", "pregnant", "domestic", "son",
             "daughter", "brother", "sister", "sibling", "offspring", "infant",
             "maternal", "paternal", "uncle", "aunt", "cousin", "relative",
             "kinship", "household", "nursery", "heir", "orphan", "guardian"),
  
  duty_obligation = c("duty", "must", "should", "expected", "obligation", 
                      "responsible", "responsibility", "obey", "comply",
                      "propriety", "proper", "ought", "bound", "submit",
                      "submission", "obedient", "obedience", "compelled",
                      "necessity", "required", "behave", "conduct", "decorum"),
  
  emotions = c("love", "emotion", "feeling", "cry", "tears", "sensitive", 
               "emotional", "affection", "tender", "gentle", "kind",
               "passion", "passionate", "heart", "sentiment", "sentimental",
               "compassion", "sympathy", "empathy", "devotion", "devoted",
               "warmth", "caring", "fond", "fondness", "adore", "cherish",
               "melancholy", "grief", "sorrow", "weep", "wept", "joy"),
  
  wealth_money = c("wealth", "money", "fortune", "income", "salary", "rich", 
                   "poor", "financial", "inheritance", "dowry", "property",
                   "estate", "pounds", "shilling", "guinea", "capital",
                   "affluent", "prosperity", "prosperous", "poverty", "debt",
                   "creditor", "tenant", "rent", "lease", "assets", "means",
                   "wealthy", "impoverished", "bankrupt", "economy", "sum"),
  
  class_society = c("class", "society", "social", "status", "rank", "gentleman", 
                    "lady", "proper", "respectable", "reputation",
                    "nobility", "noble", "aristocrat", "gentry", "upper",
                    "lower", "servant", "master", "mistress", "station",
                    "standing", "position", "breeding", "manners", "etiquette",
                    "elite", "refined", "vulgar", "common", "distinguished"),
  
  scandal_reputation = c("scandal", "reputation", "gossip", "shame", "disgrace", 
                         "improper", "rumor", "talk", "whisper",
                         "dishonor", "dishonour", "infamous", "infamy", "slander",
                         "indiscretion", "unseemly", "unbecoming", "vice",
                         "virtue", "virtuous", "modest", "modesty", "chaste",
                         "chastity", "honor", "honour", "propriety", "prudence"),
  
  career_profession = c("career", "profession", "job", "work", "business", 
                        "employment", "occupation", "trade", "doctor", 
                        "lawyer", "teacher", "nurse", "clergyman", "minister",
                        "physician", "surgeon", "clerk", "merchant", "tradesman",
                        "governess", "companion", "tutor", "apprentice",
                        "artisan", "craftsman", "laborer", "labourer", "factory",
                        "shopkeeper", "landlord", "farmer", "mill"),
  
  appearance = c("beautiful", "pretty", "handsome", "attractive", "looks", 
                 "appearance", "dress", "fashion", "elegant",
                 "lovely", "charming", "fair", "beauty", "complexion",
                 "countenance", "figure", "form", "graceful", "grace",
                 "plain", "ugly", "striking", "delicate", "fine",
                 "gown", "bonnet", "attire", "adorned", "fashionable"),
  
  power_authority = c("power", "authority", "control", "command", "lead", 
                      "decision", "independence", "freedom", "autonomy",
                      "master", "mistress", "dominion", "rule", "govern",
                      "influence", "strength", "will", "liberty", "rights",
                      "submit", "submission", "dependent", "dependence",
                      "sovereign", "agency", "self", "subject", "subservient"),
  
  education = c("education", "school", "university", "study", "read", 
                "learn", "intelligent", "clever", "wise", "knowledge",
                "educated", "scholar", "learning", "books", "reading",
                "accomplish", "accomplished", "accomplishments", "literary",
                "cultivated", "cultivation", "intellect", "intellectual",
                "ignorant", "ignorance", "tutored", "lessons", "instruction"),
  
  domesticity = c("home", "house", "household", "cooking", "cleaning", 
                  "domestic", "homemaker", "housewife", "parlour", "parlor",
                  "drawing-room", "kitchen", "hearth", "fireside", "chamber",
                  "dwelling", "residence", "abode", "housekeeper", "housekeeping",
                  "needle", "sewing", "embroidery", "needlework", "comfort",
                  "comfort", "tea", "dinner", "breakfast", "domestic duties")
)



read_data_file <- function(filepath) {
  ext <- tools::file_ext(filepath)
  if (ext %in% c("csv", "txt")) {
    return(read_csv(filepath, show_col_types = FALSE))
  } else if (ext %in% c("xlsx", "xls")) {
    return(read_excel(filepath))
  } else {
    stop("Unsupported file format. Use CSV or Excel files.")
  }
}

analyze_text_for_concepts <- function(text, concepts_list) {
  if (is.na(text)) return(FALSE)
  text_lower <- tolower(text)
  results <- any(str_detect(text_lower, concepts_list))
  return(results)
}

extract_matching_words <- function(text, concept_words) {
  if (is.na(text)) return("")
  text_lower <- tolower(text)
  matches <- concept_words[str_detect(text_lower, concept_words)]
  return(paste(matches, collapse = ", "))
}

transform_to_long_format <- function(data) {
  female_data <- data %>%
    select(female_prompt, female_result) %>%
    rename(prompt = female_prompt, result = female_result) %>%
    mutate(gender = "female")
  
  male_data <- data %>%
    select(male_prompt, male_result) %>%
    rename(prompt = male_prompt, result = male_result) %>%
    mutate(gender = "male")
  
  combined <- bind_rows(female_data, male_data) %>%
    filter(!is.na(result) & result != "")
  
  return(combined)
}

analyze_results <- function(data) {
  

  for (concept_name in names(concepts)) {
    data[[concept_name]] <- map_lgl(
      data$result, 
      ~analyze_text_for_concepts(.x, concepts[[concept_name]])
    )
  }
  

  for (concept_name in names(concepts)) {
    data[[paste0(concept_name, "_words")]] <- map_chr(
      data$result,
      ~extract_matching_words(.x, concepts[[concept_name]])
    )
  }
  
  return(data)
}

create_gender_summary <- function(analyzed_data) {
  
  concept_cols <- names(concepts)
  
  summary <- analyzed_data %>%
    group_by(gender) %>%
    summarise(
      total_responses = n(),
      across(all_of(concept_cols), ~sum(.x, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    pivot_longer(
      cols = all_of(concept_cols),
      names_to = "concept",
      values_to = "count"
    ) %>%
    group_by(concept) %>%
    mutate(
      percentage = round(count / total_responses * 100, 1),
      diff = if_else(gender == "female", 
                     percentage - percentage[gender == "male"],
                     percentage - percentage[gender == "female"])
    ) %>%
    ungroup()
  
  return(summary)
}


create_detailed_concept_analysis <- function(analyzed_data) {
  
  concept_cols <- names(concepts)
  
  details <- map_dfr(concept_cols, function(concept) {
    analyzed_data %>%
      filter(.data[[concept]] == TRUE) %>%
      select(gender, prompt, result, matches(paste0(concept, "_words"))) %>%
      mutate(concept = concept)
  }) %>%
    arrange(concept, gender)  
  
  return(details)
}


create_concept_separated_files <- function(analyzed_data, output_dir) {
  
  concept_cols <- names(concepts)
  
  for (concept in concept_cols) {
    
    concept_data <- analyzed_data %>%
      filter(.data[[concept]] == TRUE) %>%
      select(gender, prompt, result, matches(paste0(concept, "_words"))) %>%
      arrange(gender)
    
    if (nrow(concept_data) > 0) {
      
      write_csv(concept_data, 
                file.path(output_dir, paste0("concept_", concept, ".csv")))
      
      cat("Konzept:", concept, "- Gefunden:", nrow(concept_data), "Responses\n")
    }
  }
}


plot_gender_comparison <- function(summary_data) {
  
 
  diff_summary <- summary_data %>%
    filter(gender == "female") %>%
    select(concept, diff) %>%
    distinct()
  
  plot_data <- summary_data %>%
    left_join(diff_summary, by = "concept")
  
  ggplot(plot_data, aes(x = reorder(concept, diff.y), 
                        y = percentage, 
                        fill = gender)) +
    geom_col(position = "dodge", width = 0.7) +
    coord_flip() +
    labs(
      title = "Gender Bias in LLM Completions",
      subtitle = "Percentage of responses containing each concept",
      x = "Concept",
      y = "Percentage (%)",
      fill = "Gender"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom") +
    scale_fill_manual(values = c("female" = "#FFAEB9", "male" = "lightskyblue1"))
}


analyze_bias_workflow <- function(data_dir = ".", output_dir = "analysis_results") {
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  

  csv_files <- list.files(path = data_dir, 
                          pattern = "generated_test_results_.*\\.csv$", 
                          full.names = TRUE)
  
  if (length(csv_files) == 0) {
    stop("Keine CSV-Dateien mit dem Muster 'generated_test_results_*.csv' gefunden!")
  }
  
  cat("Gefundene Dateien:\n")
  print(csv_files)
  cat("\n")
  
  all_results <- list()

  
  for (file in csv_files) {
    cat("Verarbeite:", basename(file), "\n")
    
    
    data <- read_data_file(file)
    
   
    long_data <- transform_to_long_format(data)
    
    analyzed <- analyze_results(long_data)
    
    filename <- tools::file_path_sans_ext(basename(file))
    write_csv(analyzed, file.path(output_dir, 
                                  paste0(filename, "_analyzed.csv")))
    
    all_results[[filename]] <- analyzed
  }
  
  combined <- bind_rows(all_results, .id = "source_file")
  write_csv(combined, file.path(output_dir, "all_results_combined.csv"))
  
  summary <- create_gender_summary(combined)
  write_csv(summary, file.path(output_dir, "gender_summary.csv"))
  
  detailed <- create_detailed_concept_analysis(combined)
  write_csv(detailed, file.path(output_dir, "detailed_concept_examples.csv"))
  
  cat("\nErstelle separate Dateien pro Konzept...\n")
  create_concept_separated_files(combined, output_dir)
  
  diff_table <- summary %>%
    select(gender, concept, percentage) %>%
    pivot_wider(names_from = gender, values_from = percentage) %>%
    mutate(difference = female - male) %>%
    arrange(desc(abs(difference)))
  
  write_csv(diff_table, file.path(output_dir, "gender_difference_ranking.csv"))
  
  #Plot
  plot <- plot_gender_comparison(summary)
  ggsave(file.path(output_dir, "gender_comparison.png"), 
         plot, width = 12, height = 8, dpi = 300)
  
  #Zusammenfassung
  cat("\n===== ANALYSE ABGESCHLOSSEN =====\n")
  cat("Gesamtzahl Responses:", nrow(combined), "\n")
  cat("Female Responses:", sum(combined$gender == "female"), "\n")
  cat("Male Responses:", sum(combined$gender == "male"), "\n")
  cat("\nTop 5 Unterschiede (Female % - Male %):\n")
  print(head(diff_table, 5))
  cat("\nAlle Ergebnisse gespeichert in:", output_dir, "\n")
  
  return(list(
    combined_data = combined,
    summary = summary,
    detailed = detailed,
    difference_ranking = diff_table,
    plot = plot
  ))
}

