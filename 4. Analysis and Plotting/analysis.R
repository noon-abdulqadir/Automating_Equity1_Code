# ### Setup

options("install.packages.compile.from.source" = "never")
file.path(R.home("bin"), "R")
# chooseCRANmirror(ind=58)
# install.packages(c("knitr", "tidyverse", "plyr", "dplyr", "purrr", "data.table", "DT", "jtools", "PMCMRplus", "glue", "stargazer", "ggstatsplot", "ggplot2", "ggpubr", "rstatix", "sjPlot", "lattice", "car", "lme4", "lmeInfo", "multiverse", "specr", "texreg", "performance", "broom", "broom.mixed", "AICcmodavg", "reticulate"))

## ----Import libraries, message=TRUE, warning=FALSE, paged.print=TRUE----------
# Load libraries
# chooseCRANmirror(ind=58)
library(knitr)
library(tidyverse)
library(plyr)
library(dplyr)
library(purrr)
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
# library(citr)


## ----Set.Python, message=TRUE, warning=FALSE, paged.print=TRUE----------------
## Set Python
code_dir <- ''
# setwd("..")
# code_dir <- getwd()
reticulate::use_condaenv('/opt/homebrew/Caskroom/miniforge/base/envs/study1')
reticulate::use_python('/opt/homebrew/Caskroom/miniforge/base/envs/study1/bin/python3.8', required = TRUE)
reticulate::source_python(file = glue('{code_dir}/setup_module/params.py'), envir = globalenv(), convert = TRUE)
pd <- import("pandas")
set.seed(42)


## ----Read dataframes, message=TRUE, warning=FALSE, paged.print=TRUE-----------
## Read dfset with outliers removed
## READ PICKLE
if (analysis_df_from_manual is True) {
  df_name = 'df_manual'
  df_file = 'outliers'

} else if (analysis_df_from_manual is False) {
  df_name = 'df'
  df_file = glue('outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}')

}

df <- pd$read_pickle(glue("{df_dir}{df_name}_{df_file}.{file_save_format}"))
df_mean <- pd$read_pickle(glue("{df_dir}{df_name}_mean_{df_file}.{file_save_format}"))
# dataframes[[1]] <- pd$read_pickle(glue("{df_dir}{df_name}_{df_file}.{file_save_format}"))
# dataframes[[2]] <- pd$read_pickle(glue("{df_dir}{df_name}_mean_{df_file}.{file_save_format}"))



## ----Make mean df, message=TRUE, warning=FALSE, paged.print=TRUE--------------
## Make mean df
if (!exists("dataframes[[2]]")) {
  print("DF MEAN DOES NOT EXIST. MAKING DF.")
  dataframes[[2]] = ddply(
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
} else if (exists("dataframes[[2]]")) {
  print(glue("DF MEAN ALREADY EXISTS."))
  print(glue("USING DF FROM FILE: {df_dir}{df_name}_mean_outliers_age_limit-{age_limit}_age_ratio-{age_ratio}_gender_ratio-{gender_ratio}.pkl"))
}



## ----Dataframe overview, message=TRUE, warning=FALSE, paged.print=TRUE--------
df_names <- names(df) %>% as.data.frame()
colnames(df_names) <- c("Variable Names")

DT::datatable(df_names)


## ----MEAN Dataframe overview, message=TRUE, warning=FALSE, paged.print=TRUE----
df_mean_names <- names(df_mean) %>% as.data.frame()
colnames(df_mean_names) <- c("MEAN Variable Names")

DT::datatable(df_mean_names)


## ----Summary, message=TRUE, warning=FALSE, paged.print=TRUE-------------------
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


## ----Set factors, message=TRUE, warning=FALSE, paged.print=TRUE---------------
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


## ----Set variables, message=TRUE, warning=FALSE, paged.print=TRUE-------------
data <- df_mean
gender_iv = "Gender"
age_iv = "Age"
outliers_remove <- FALSE
for (dv in dv_cols){
  if (outliers_remove is False){
      assign(paste(glue('{tolower(dv)}'), 'dv', sep = "_"), dv)
      assign(paste(glue('{tolower(dv)}'), 'proba', sep = "_"), paste(dv, 'Probability', sep = "_"))

    } else if (outliers_remove is True){
      assign(paste(glue('{tolower(dv)}'), 'dv', sep = "_"), paste(dv, 'Outliers_Removed', sep = "_"))
      assign(paste(glue('{tolower(dv)}'), 'proba', sep = "_"), paste(dv, 'Probability_Outliers_Removed', sep = "_"))
  }
}

for (iv in ivs){
  for (dv in dv_cols){
    assign(paste(glue('{tolower(iv)}'), glue('{tolower(dv)}_mean'), sep = "_"), aggregate(data[[dv]], list(data[[iv]]), FUN=mean))

  }
}


## ----Visualize, message=TRUE, warning=FALSE, paged.print=TRUE-----------------
for (dv in dv_cols){

  strrep("=",80)
  print(glue('Density plot for {iv} x {dv}:'))
  density_plot = ggdensity(data,
                            main = glue("Density plot of {dv}"),
                            x = dv,
                            xlab = glue("{dv}"),
                            color = 'black',
                            fill = "lightgray",
                            rug = TRUE,
                            inherit.aes = TRUE,
                            ggtheme = theme_minimal()) +
  stat_central_tendency(type = "mean", color = "red", linetype = 2, show.legend = TRUE)
  # stat_central_tendency(type = "median", color = "blue", linetype = 2, show.legend = TRUE)

  print(density_plot)
  #### Save density plot
  ggplot2::ggsave(
      filename = glue("{plot_save_path}Densityplot {df_name} - {dv}.png"),
      plot = density_plot,
      device = "png",
      dpi = 1200,
      width = 15,
      height = 10,
      units = "cm"
  )

  for (iv in ivs){
    strrep("=",80)
    print(glue('Density plot for {iv} x {dv}:'))
    density_plot = ggdensity(data,
                              main = glue("Density plot of {iv} x {dv}"),
                              x = dv,
                              xlab = glue("{dv}"),
                              facet.by = iv,
                              color = 'black',
                              fill = "lightgray",
                              rug = TRUE,
                              inherit.aes = TRUE,
                              ggtheme = theme_minimal()) +
    stat_central_tendency(type = "mean", color = "red", linetype = 2, show.legend = TRUE)
    # stat_central_tendency(type = "median", color = "blue", linetype = 2, show.legend = TRUE)

    print(density_plot)
    #### Save density plot
    ggplot2::ggsave(
        filename = glue("{plot_save_path}Densityplot {df_name} - {iv} x {dv}.png"),
        plot = density_plot,
        device = "png",
        dpi = 1200,
        width = 15,
        height = 10,
        units = "cm"
    )
  }
}



## ----Function to perform analysis, message=TRUE, warning=FALSE, paged.print=TRUE----
## Function to perform analysis
analysis_func <- function(df, iv, dv){
  strrep("=",80)
  print(glue('Analyzing {df_name}'))
  strrep("-",20)
  print(glue('{iv} x {dv}'))
  strrep("-",20)
  ## Levene's Test
  lev = leveneTest(data = df, data[[dv]] ~ data[[iv]])

  if (lev["group", 3] <= 0.05){
      lev_not_sig = FALSE
      print(glue("Levene's test is NOT significant at {lev['group', 3]}"))
      } else if (lev["group", 3] >= 0.05){
      lev_not_sig = TRUE
      print(glue("Levene's test is significant at {lev['group', 3]}"))
  }

  ## One-way Welch's ANOVA
  strrep("-",20)
  print(glue("One-way Welch's ANOVA for {iv} x {dv}"))
  strrep("-",20)
  one_way <-
      aov(data[[dv]] ~ as.factor(data[[iv]]),
          data = df,
          var.equal = lev_not_sig)
  anova(one_way)
  res <- gamesHowellTest(one_way)
  summaryGroup(res)
  summary(res)

  ## OLS Regression
  strrep("-",20)
  print(glue('Regression for {iv} x {dv}'))
  strrep("-",20)
  lm <- lm(data[[dv]] ~ as.factor(data[[iv]]), data = df)
  summ(lm)
  summary(lm)$coef
  par(mfrow = c(2, 2))
  plot(lm)
  return(lev_not_sig)
  strrep("=",80)

}


## ----ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----
for (iv in ivs){
  for (dv in dv_cols){
    lev_not_sig <- analysis_func(df = data,
                                 iv = iv,
                                 dv = dv)
    vplot <- ggbetweenstats(
      data = data,
      x = data[[iv]],
      y = data[[dv]],
      xlab = glue("{iv} segregated sectors"),
      ylab = glue("Presence of {dv}-related frames"),
      type = "parametric",
      conf.level = 0.95,
      # ANOVA or Kruskal-Wallis
      var.equal = lev_not_sig,
      # ANOVA or Welch ANOVA
      plot.type = "boxviolin",
      mean.plotting = TRUE,
      outlier.tagging = TRUE,
      outlier.coef = 1.5,
      outlier.label = region,
      outlier.label.color = "red",
      sphericity.correction = TRUE,
      p.adjust.method = "bonferroni",
      pairwise.comparisons = TRUE,
      pairwise.display = "significant",
      centrality.plotting = TRUE,
      centrality.path = TRUE,
      centrality.type = "parameteric",
      bf.message = TRUE,
      title = glue("Violin plot of {dv}-related frames in job ads from {iv} segregated sectors"),
      caption = glue("{dv}-{iv} Violin plot ")
    )
    print(vplot)

    # #### Save violin plot
    # ggplot2::ggsave(
    #     filename = glue("{plot_save_path}Violinplot {df_name} - {iv} x {dv}.png"),
    #     plot = vplot,
    #     device = "png",
    #     dpi = 1200,
    #     width = 15,
    #     height = 10,
    #     units = "cm"
    # )
  }
}


## ----Gender-Warmth ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----
#### Gender-Warmth
warm_gen_lev_not_sig <- analysis_func(iv = data$gender,
                                    dv = data$warmth,
                                    df = data)


## ----Gender-Warmth Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE----
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
    var.equal = lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    mean.plotting = TRUE,
    outlier.tagging = TRUE,
    outlier.coef = 1.5,
    outlier.label = region,
    outlier.label.color = "red",
    sphericity.correction = TRUE,
    p.adjust.method = "bonferroni",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    centrality.path = TRUE,
    centrality.type = "parameteric",
    bf.message = TRUE,
    title = glue("Violin plot of {warmth_dv}-related frames in job ads from {gender_iv} segregated sectors"),
    caption = glue("{warmth_dv}-{gender_iv} Violin plot ")
)
print(warm_gen_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violinplot {df_name} - {gender_iv} x {warmth_dv}.png"),
    plot = warm_gen_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Gender-Competence ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----
#### Gender-Competence
comp_gen_lev_not_sig <- analysis_func(iv = data$Gender,
                                    dv = data$Competence,
                                    df = data)


## ----Gender-Competence Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE----
## Gender-Competence Violin plot
comp_gen_vplot <- ggbetweenstats(
    data = data,
    x = Gender,
    y = Competence,
    xlab = glue("{gender_iv} segregated sectors"),
    ylab = glue("Presence of {competence_dv}-related frames"),
    type = "parametric",
    conf.level = 0.95,
    # ANOVA or Kruskal-Wallis
    var.equal = lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    mean.plotting = TRUE,
    outlier.tagging = TRUE,
    outlier.coef = 1.5,
    outlier.label = region,
    outlier.label.color = "red",
    sphericity.correction = TRUE,
    p.adjust.method = "bonferroni",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    centrality.path = TRUE,
    centrality.type = "parameteric",
    bf.message = TRUE,
    title = glue("Violin plot of {competence_dv}-related frames in job ads from {gender_iv} segregated sectors"),
    caption = glue("{competence_dv}-{gender_iv} Violin plot")
)
print(comp_gen_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violinplot {df_name} - {gender_iv} x {competence_dv}.png"),
    plot = comp_gen_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Age-Warmth ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----
#### Age-Warmth
warm_age_lev_not_sig <- analysis_func(iv = data$Age,
                                    dv = data$Warmth,
                                    df = data)


## ----Age-Warmth Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE----
## Age-Warmth Violin plot
warm_age_vplot <- ggbetweenstats(
    data = data,
    x = Age,
    y = Warmth,
    xlab = glue("{age_iv} segregated sectors"),
    ylab = glue("Presence of {warmth_dv}-related frames"),
    type = "parametric",
    conf.level = 0.95,
    # ANOVA or Kruskal-Wallis
    var.equal = lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    mean.plotting = TRUE,
    outlier.tagging = TRUE,
    outlier.coef = 1.5,
    outlier.label = region,
    outlier.label.color = "red",
    sphericity.correction = TRUE,
    p.adjust.method = "bonferroni",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    centrality.path = TRUE,
    centrality.type = "parameteric",
    bf.message = TRUE,
    title = glue("Violin plot of {warmth_dv}-related frames in job ads from {age_iv} segregated sectors"),
    caption = glue("{warmth_dv}-{age_iv} Violin plot")
)
print(warm_age_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violinplot {df_name} - {age_iv} x {warmth_dv}.png"),
    plot = warm_age_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Age-Competence ANOVA and OLS regression, message=TRUE, warning=FALSE, paged.print=TRUE----
#### Age-Competence
comp_age_lev_not_sig <- analysis_func(iv = data$Age,
                                    dv = data$Competence,
                                    df = data)


## ----Age-Competence Violin plot, message=TRUE, warning=FALSE, paged.print=TRUE----
## Age-Competence Violin plot
comp_age_vplot <- ggbetweenstats(
    data = data,
    x = Age,
    y = Competence,
    xlab = glue("{age_iv} segregated sectors"),
    ylab = glue("Presence of {competence_dv}-related frames"),
    type = "parametric",
    conf.level = 0.95,
    # ANOVA or Kruskal-Wallis
    var.equal = lev_not_sig,
    # ANOVA or Welch ANOVA
    plot.type = "boxviolin",
    mean.plotting = TRUE,
    outlier.tagging = TRUE,
    outlier.coef = 1.5,
    outlier.label = region,
    outlier.label.color = "red",
    sphericity.correction = TRUE,
    p.adjust.method = "bonferroni",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    centrality.plotting = TRUE,
    centrality.path = TRUE,
    centrality.type = "parameteric",
    bf.message = TRUE,
    title = glue("Violin plot of {competence_dv}-related frames in job ads from {age_iv} segregated sectors"),
    caption = glue("Competence-{age_iv} Violin plot")
)
print(comp_age_vplot)

#### Save violin plot
ggplot2::ggsave(
    filename = glue("{plot_save_path}Violinplot {df_name} - {age_iv} x {competence_dv}.png"),
    plot = comp_age_vplot,
    device = "png",
    dpi = 1200,
    width = 15,
    height = 10,
    units = "cm"
)


## ----Multi-levl Log Reg, message=TRUE, warning=FALSE, paged.print=TRUE--------
library(sjstats)
lg_model_w <- glmer(warmth ~ (1 | job_id), data = df, family =binomial)
summary(lg_model_w)
icc(lg_model_w)

lvl1_lg_model_wg <- glmer(warmth ~ gender + (1 | job_id), data = df, family =binomial)
summary(lvl1_lg_model_wg)

# lg_model <- glmer(competence ~ (1 | job_id), data = df, family =binomial)
# summary(lg_model)
# icc(lg_model)
#
# lvl1_lg_model <- glmer(competence ~ age + (1 | job_id), data = df, family =binomial)
# summary(lvl1_lg_model)

# lg_model <- glmer(warmth ~ (1 | job_id), data = df, family =binomial)
# summary(lg_model)
# icc(lg_model)

lvl1_lg_model_wa <- glmer(warmth ~ age + (1 | job_id), data = df, family =binomial)
summary(lvl1_lg_model_wa)

lvl1_lg_model_wag <- glmer(warmth ~ age + gender + (1 | job_id), data = df, family =binomial)
summary(lvl1_lg_model_wag)

# lg_model <- glmer(competence ~ (1 | job_id), data = df, family =binomial)
# summary(lg_model)
# icc(lg_model)
#
# lvl1_lg_model <- glmer(competence ~ age + (1 | job_id), data = df, family =binomial)
# summary(lvl1_lg_model)
# icc(lg_model)
#
test_lr = test_likelihoodratio(lg_model_w, lvl1_lg_model_wg, lvl1_lg_model_wag)




## -----------------------------------------------------------------------------
# Calculate odds ration
library(sjPlot)

get_model_data(lvl1_lg_model_wag, type = 'std')

# effectsize::standardize_parameters(lvl1_lg_model_wag, exponentate = TRUE, method='basic')



# # -----------------------------------------------------------------------------

interpret_oddsratio(0.3955740)


## -----------------------------------------------------------------------------
plot_model(lvl1_lg_model_wag, type = 'eff')
# BE


## -----------------------------------------------------------------------------
plot_model(lvl1_lg_model_wg, type = 'eff')


## -----------------------------------------------------------------------------
lg_model_w <- glmer(warmth ~ (1 | job_id), data = df, family =binomial)
summary(lg_model_w)
icc(lg_model_w)

lvl1_lg_model_wg <- glmer(warmth ~ gender +  num_chars_log + (1 | job_id), data = df, family =binomial)
summary(lvl1_lg_model_wg)

lvl1_lg_model_wa <- glmer(warmth ~ age + (1 | job_id) + num_chars_log, data = df, family =binomial)
summary(lvl1_lg_model_wa)

lvl1_lg_model_wag <- glmer(warmth ~ age + gender + (1 | job_id) + num_chars_log, data = df, family =binomial)
summary(lvl1_lg_model_wag)


## -----------------------------------------------------------------------------
lvl1_lg_model_wag_int <- glmer(warmth ~ age * gender + (1 | job_id), data = df, family =binomial)
summary(lvl1_lg_model_wag_int)


## -----------------------------------------------------------------------------
test_lr = test_likelihoodratio(lg_model_w, lvl1_lg_model_wg, lvl1_lg_model_wag)

print(test_lr)



## -----------------------------------------------------------------------------
performance::check_model(lvl1_lg_model_wg)


## -----------------------------------------------------------------------------
hist(log(df$num_chars))

# TEll about directionality but not effect size
df$num_chars_log <- log(df$num_chars)


# # ----Null model, message=TRUE, warning=FALSE, paged.print=TRUE----------------

library(janitor)
# df <- janitor::clean_names(df)
null_model <- lmer(warmth ~ (1 | job_id), data = df)
summary(null_model)


## ----First level model, message=TRUE, warning=FALSE, paged.print=TRUE---------
lvl1_preds_model <- lmer(warmth ~ gender + (1 | job_id), data = df)
summary(lvl1_preds_model)


## ----Check performance of first level model, message=TRUE, warning=FALSE, paged.print=TRUE----
performance::check_model(lvl1_preds_model)


## ----Check first level model parameters, message=TRUE, warning=FALSE, paged.print=TRUE----
broom.mixed::tidy(lvl1_preds_model)


## ----Plot first level model, message=TRUE, warning=FALSE, paged.print=TRUE----
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


## ----Plot first level model coefficients, message=TRUE, warning=FALSE, paged.print=TRUE----
sjPlot::plot_model(lvl1_preds_model, show.p = T, show.values = T)


## ----Comparing the null model and the first level model, message=TRUE, warning=FALSE, paged.print=TRUE----
htmltools::HTML(htmlreg(list(null_model, lvl1_preds_model)))


## ----Random slop model, message=TRUE, warning=FALSE, paged.print=TRUE---------
rs_preds_model <- lmer(Warmth ~ Gender + (Gender | Job.ID), data = df)
summary(rs_preds_model)


## ----Check performance of random slope model, message=TRUE, warning=FALSE, paged.print=TRUE----
performance::check_model(rs_preds_model)


## ----Check random slope model parameters, message=TRUE, warning=FALSE, paged.print=TRUE----
broom.mixed::tidy(rs_preds_model)


## ----Plot random slope model, message=TRUE, warning=FALSE, paged.print=TRUE----
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


## ----Plot random slope model coefficients, message=TRUE, warning=FALSE, paged.print=TRUE----
sjPlot::plot_model(rs_preds_model, show.p = T, show.values = T)


## ----Comparing the null model, the first level model, and the random slope model, message=TRUE, warning=FALSE, paged.print=TRUE----
htmltools::HTML(htmlreg(list(null_model, lvl1_preds_model, rs_preds_model)))


## ----ANOVA to compare models, message=TRUE, warning=FALSE, paged.print=TRUE----
anova(null_model, lvl1_preds_model, rs_preds_model)


## -----------------------------------------------------------------------------
spec_curve <- run_specs(df = df,
                        y = c("Warmth", "Competence"),
                        x = c("Gender", "Age"),
                        model = c("lm"))
head(spec_curve)


## -----------------------------------------------------------------------------
plot_specs(spec_curve, choices = c("x", "y"))


## -----------------------------------------------------------------------------
plot_decisiontree(spec_curve, legend = TRUE, label = T)


## ----Save as .r file, message=TRUE, warning=FALSE, paged.print=TRUE-----------
knitr::purl(glue("{code_dir}/Analysis/analysis.Rmd"))
