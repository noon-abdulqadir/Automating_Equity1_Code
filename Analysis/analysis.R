## ----Import libraries, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------------------------
# Load libraries
library(knitr)
# library(tidyverse)
library(plyr)
# library(dplyr)
library(data.table)
library(DT)
library(jtools)
library(PMCMRplus)
library(glue)
library(stargazer)
library(ggstatsplot)
library(ggplot2)
library(ggpubr)
library(rstatix)
library(sjPlot)
library(lattice)
library(car)
library(lme4)
library(lmeInfo)
library(multiverse)
library(specr)
# library(rdfanalysis)
library(texreg)
library(performance)
library(broom)
library(broom.mixed)
library(AICcmodavg)
library(reticulate)

## Set Python
use_python("/opt/homebrew/Caskroom/miniforge/base/envs/study1/bin/python")
source_python(glue('{code_dir}/setup_module/params.py'))
pd <- import("pandas")



## ----Read dataframes, message=TRUE, warning=FALSE, paged.print=TRUE---------------------------------------------------
## Read dfset with outliers removed
## READ PICKLE
if (analysis_df_from_manual == TRUE) {
  df_name = 'df_manual'
  df <-
    pd$read_pickle(glue("{df_dir}{df_name}_outliers.pkl"))

  df_mean <-
    pd$read_pickle(glue("{df_dir}{df_name}_mean_outliers.pkl"))

} else if (analysis_df_from_manual == FALSE) {
  df_name = 'df'
  df <-
    pd$read_pickle(glue("{df_dir}{df_name}_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.pkl"))

  df_mean <-
    pd$read_pickle(glue("{df_dir}{df_name}_mean_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.pkl"))

}

# ## READ CSV
# if (analysis_df_from_manual == TRUE) {
#   df_name = 'df_manual'
#   df <-
#     read.csv(
#     glue("{df_dir}{df_name}_outliers.csv"),
#     header = TRUE
#     )
#
#   df_mean <-
#     read.csv(
#     glue("{df_dir}{df_name}_mean_outliers.csv"),
#     header = TRUE
#     )
#
# } else if (analysis_df_from_manual == FALSE) {
#   df_name = 'df'
#   df <-
#     read.csv(
#       glue("{df_dir}{df_name}_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.csv"),
#     header = TRUE
#     )
#
#   df_mean <-
#     read.csv(
#       glue("{df_dir}{df_name}_mean_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.csv"),
#     header = TRUE
#     )
#
# }


## ----Make mean df, message=TRUE, warning=FALSE, paged.print=TRUE------------------------------------------------------
## Make mean df
if (!exists("df_mean")) {
  print("DF MEAN DOES NOT EXIST. MAKING DF.")
  df_mean = ddply(
    df,
    .(Job.ID),
    summarise,
    # Search.Keyword=Search.Keyword[1],
    Warmth = mean(Warmth),
    Competence = mean(Competence),
    # Warmth_Probability = mean(Warmth_Probability),
    # Competence_Probability = mean(Competence_Probability),
    Gender=Gender[1],
    Age=Age[1],
    Gender_Female=Gender_Female[1],
    Gender_Male=Gender_Male[1],
    Gender_Mixed=Gender_Mixed[1],
    Age_Older=Age_Older[1],
    Age_Younger=Age_Younger[1],
    Age_Mixed=Age_Mixed[1],
    Gender_Num=Gender_Num[1],
    Age_Num=Age_Num[1]
    # Collection.Date=Collection.Date[1]
)
} else if (exists("df_mean")) {
  print(glue("DF MEAN ALREADY EXISTS."))
  print(glue("USING DF FROM FILE: {df_dir}{df_name}_mean_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.pkl"))
}



## ----Dataframe overview, message=TRUE, warning=FALSE, paged.print=TRUE------------------------------------------------
df_names <- names(df) %>% as.data.frame()
colnames(df_names) <- c("Variable Names")

DT::datatable(df_names)


## ----MEAN Dataframe overview, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------
df_mean_names <- names(df_mean) %>% as.data.frame()
colnames(df_mean_names) <- c("MEAN Variable Names")

DT::datatable(df_mean_names)


## ----Summary, message=TRUE, warning=FALSE, paged.print=TRUE-----------------------------------------------------------
## df descriptives
strrep("=",80)
print(glue("DF of length {nrow(df)}:"))
strrep("-",20)
summary(df[c(ivs_all, dv_cols)])
strrep("=",80)
print(glue("DF MEAN of length {nrow(df_mean)}:"))
strrep("-",20)
summary(df_mean[c(ivs_all, dv_cols)])
strrep("=",80)


## ----Set factors, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------------------
## Set factors
# DF
df$Gender <-
    factor(df$Gender, levels = order_gender)
attributes(df$Gender)
df$Age <-
    factor(df$Age, levels = order_age)
attributes(df$Age)

# DF MEAN
df_mean$Gender <-
    factor(df_mean$Gender, levels = order_gender)
attributes(df_mean$Gender)
df_mean$Age <-
    factor(df_mean$Age, levels = order_age)
attributes(df_mean$Age)


## ----Set variables, message=TRUE, warning=FALSE, paged.print=TRUE-----------------------------------------------------
data <- df_mean
gender_iv = "Gender"
age_iv = "Age"
outliers_remove <- FALSE
if (outliers_remove == FALSE){
    warmth_dv = 'Warmth'
    warmth_proba = 'Warmth_Probability'
    competence_dv = 'Competence'
    competence_proba = 'Competence_Probability'
    } else if (outliers_remove == TRUE){
    warmth_dv = 'Warmth_Outliers_Removed'
    warmth_proba = 'Warmth_Probability_Outliers_Removed'
    competence_dv = 'Competence_Outliers_Removed'
    competence_proba = 'Competence_Probability_Outliers_Removed'
}
gender_warm_mean = aggregate(data$Warmth, list(data$Gender), FUN=mean)
gender_comp_mean = aggregate(data$Competence, list(data$Gender), FUN=mean)
age_warm_mean = aggregate(data$Warmth, list(data$Age), FUN=mean)
age_comp_mean = aggregate(data$Competence, list(data$Age), FUN=mean)


## ----Function to perform analysis, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------------
## Function to perform analysis
analysis_func <- function(iv, dv, df){
    ## Warmth-Gender Levene's Test
    lev = leveneTest(dv ~ iv, data = df)

    if (lev["group", 3] <= 0.05){
        lev_not_sig = FALSE
        } else if (lev["group", 3] >= 0.05){
        lev_not_sig = TRUE
    }

    ## Warmth-Gender One-way Welch's ANOVA
    one.way <-
        aov(dv ~ as.factor(iv),
            data = df,
            var.equal = lev_not_sig)
    anova(one.way)
    res <- gamesHowellTest(one.way)
    summaryGroup(res)
    summary(res)

    ## Warmth-Gender OLS Regression
    lm <- lm(dv ~ as.factor(iv), data = df)
    summ(lm)
    summary(lm)$coef
    par(mfrow = c(2, 2))
    plot(lm)
    return(lev_not_sig)
}


## ----Gender-Warmth ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----------------------------
#### Gender-Warmth
warm_gen_lev_not_sig <- analysis_func(iv = data$Gender,
                                    dv = data$Warmth,
                                    df = data)


## ----Gender-Warmth Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE-----------------------------------------
## Gender-Warmth Violin plot
warm_gen_vplot <- ggbetweenstats(
    data = data,
    x = Gender,
    y = Warmth,
    xlab = glue("{gender_iv} segregated sectors"),
    ylab = glue("Presence of {warmth_dv}-related frames"),
    type = "parametric",
    conf.level = 0.95,
    # ANOVA or Kruskal-Wallis
    var.equal = warm_gen_lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    bf.message = FALSE,
    title = glue("Violin plot of {warmth_dv}-related frames in job ads from {gender_iv} segregated sectors"),
    caption = glue("{warmth_dv}-{gender_iv} Violin plot ")
)
print(warm_gen_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violin Plot {df_name} - {gender_iv} x {warmth_dv}.png"),
    plot = warm_gen_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Gender-Competence ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE------------------------
#### Gender-Competence
comp_gen_lev_not_sig <- analysis_func(iv = data$Gender,
                                    dv = data$Competence,
                                    df = data)


## ----Gender-Competence Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------
## Gender-Competence Violin plot
comp_gen_vplot <- ggbetweenstats(
    data = data,
    x = Gender,
    y = Competence,
    xlab = glue("{gender_iv} segregated sectors"),
    ylab = glue("Presence of {competence_dv}-related frames"),
    type = "parametric",
    # ANOVA or Kruskal-Wallis
    var.equal = comp_gen_lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    bf.message = FALSE,
    title = glue("Violin plot of {competence_dv}-related frames in job ads from {gender_iv} segregated sectors"),
    caption = glue("{competence_dv}-{gender_iv} Violin plot")
)
print(comp_gen_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violin Plot {df_name} - {gender_iv} x {competence_dv}.png"),
    plot = comp_gen_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Age-Warmth ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------
#### Age-Warmth
warm_age_lev_not_sig <- analysis_func(iv = data$Age,
                                    dv = data$Warmth,
                                    df = data)


## ----Age-Warmth Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------------------
## Age-Warmth Violin plot
warm_age_vplot <- ggbetweenstats(
    data = data,
    x = Age,
    y = Warmth,
    xlab = glue("{age_iv} segregated sectors"),
    ylab = glue("Presence of {warmth_dv}-related frames"),
    # ANOVA or Kruskal-Wallis
    var.equal = warm_age_lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    bf.message = FALSE,
    title = glue("Violin plot of {warmth_dv}-related frames in job ads from {age_iv} segregated sectors"),
    caption = glue("{warmth_dv}-{age_iv} Violin plot")
)
print(warm_age_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violin Plot {df_name} - {age_iv} x {warmth_dv}.png"),
    plot = warm_age_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Age-Competence ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE---------------------------
#### Age-Competence
comp_age_lev_not_sig <- analysis_func(iv = data$Age,
                                    dv = data$Competence,
                                    df = data)


## ----Age-Competence Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE----------------------------------------
## Age-Competence Violin plot
comp_age_vplot <- ggbetweenstats(
    data = data,
    x = Age,
    y = Competence,
    xlab = glue("{age_iv} segregated sectors"),
    ylab = glue("Presence of {competence_dv}-related frames"),
    type = "parametric",
    # ANOVA or Kruskal-Wallis
    var.equal = comp_age_lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    bf.message = FALSE,
    title = glue("Violin plot of {competence_dv}-related frames in job ads from {age_iv} segregated sectors"),
    caption = glue("Competence-{age_iv} Violin plot")
)
print(comp_age_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violin Plot {df_name} - {age_iv} x {competence_dv}.png"),
    plot = comp_age_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Save as .r file, message=TRUE, warning=FALSE, paged.print=TRUE---------------------------------------------------
knitr::purl(glue("{code_dir}/Analysis/analysis.Rmd"))


## ----Null model, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------------------------------
null_model <- lmer(Warmth_Probability ~ (1 | Job.ID), data = df)
summary(null_model)


## ----First level model, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------------
lvl1_preds_model <- lmer(Warmth_Probability ~ Gender + (1 | Job.ID), data = df)
summary(lvl1_preds_model)


## ----Check performance of first level model, message=TRUE, warning=FALSE, paged.print=TRUE----------------------------
performance::check_model(lvl1_preds_model)


## ----Check first level model parameters, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------
broom.mixed::tidy(lvl1_preds_model)


## ----Plot first level model, message=TRUE, warning=FALSE, paged.print=TRUE--------------------------------------------
broom.mixed::augment(lvl1_preds_model) %>%
    filter(Job.ID %in% Job.IDs) %>%
    ggplot(aes(x=education, y=income)) +
    geom_line(aes(x = Gender,
                    y=.fitted,
                    color = Job.ID),
                inherit.aes=FALSE, size = 1) +
    theme_minimal() +
    theme(legend.position="none") +
    ggthemes::scale_color_gdocs()


## ----Plot first level model coefficients, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------
sjPlot::plot_model(lvl1_preds_model, show.p = T, show.values = T)


## ----Comparing the null model and the first level model, message=TRUE, warning=FALSE, paged.print=TRUE----------------
htmltools::HTML(htmlreg(list(null_model, lvl1_preds_model)))


## ----Random slop model, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------------
rs_preds_model <- lmer(Warmth ~ Gender + (Gender | Job.ID), data = df)
summary(rs_preds_model)


## ----Check performance of random slope model, message=TRUE, warning=FALSE, paged.print=TRUE---------------------------
performance::check_model(rs_preds_model)


## ----Check random slope model parameters, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------
broom.mixed::tidy(rs_preds_model)


## ----Plot random slope model, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------
broom.mixed::augment(rs_preds_model) %>%
    filter(Job.ID %in% Job.IDs) %>%
    ggplot(aes(x=education, y=income)) +
    geom_line(aes(x = Gender,
                    y=.fitted,
                    color = Job.ID),
                inherit.aes=FALSE, size = 1) +
    theme_minimal() +
    theme(legend.position="none") +
    ggthemes::scale_color_gdocs()


## ----Plot random slope model coefficients, message=TRUE, warning=FALSE, paged.print=TRUE------------------------------
sjPlot::plot_model(rs_preds_model, show.p = T, show.values = T)


## ----Comparing the null model, the first level model, and the random slope model, message=TRUE, warning=FALSE, paged.print=TRUE----
htmltools::HTML(htmlreg(list(null_model, lvl1_preds_model, rs_preds_model)))


## ----ANOVA to compare models, message=TRUE, warning=FALSE, paged.print=TRUE-------------------------------------------
anova(null_model, lvl1_preds_model, rs_preds_model)


## ---------------------------------------------------------------------------------------------------------------------
spec_curve <- run_specs(df = df,
                        y = c("Warmth", "Competence"),
                        x = c("Gender", "Age"),
                        model = c("lm"))
head(spec_curve)


## ---------------------------------------------------------------------------------------------------------------------
plot_specs(spec_curve, choices = c("x", "y"))


## ---------------------------------------------------------------------------------------------------------------------
plot_decisiontree(spec_curve, legend = TRUE, label = T)
