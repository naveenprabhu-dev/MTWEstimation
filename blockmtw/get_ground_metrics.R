# Word Embeddings Transformers

# These are manually inserted into the ema_script_imputation.py file.

library(text)
library(dplyr)
rm(list = ls())
# texts_full <- c("I found it difficult to relax", 
#            "I felt (very) irritable", 
#            "I was worried about different things", 
#            "I felt nervous, anxious, or on edge",
#            "I felt that I had nothing to look forward", 
#            "I couldn't seem to experience any positive feeling at all", 
#            "I felt tired", 
#            "I felt like I lack companionship, or that I am not close to people", 
#            "I spent _ on meaningful, offline, social interaction", 
#            "I spent _ using social media to kill/pass the time", 
#            "I spent _ outside (outdoors)", 
#            "I spent _ occupied with the coronavirus (e.g., watching news, thinking about it, talking to friends about it", 
#            "I spent _ thinking about my own health or that of my close friends and family members regarding the coronavirus",
#            "I spent _ at home (including the home of parents/partner)")

texts_8 <- c("I found it difficult to relax", 
             "I felt very irritable", 
             "I was worried about different things", 
             "I felt nervous, anxious, or on edge", 
             "I spent _ minutes on meaningful, offline, social interaction",
             "I spent _ using social media to kill/pass the time", 
             "I spent _ outside (outdoors)",
             "I spent _ at home (including the home of parents/partner)")

embed_bert <- textEmbed(texts_8)

dist_bert <- textDistanceMatrix(embed_Bert$texts$texts)

embed_roberta <- textEmbed(texts_8, model="roberta-base")

dist_roberta <- textDistanceMatrix((embed_roberta$texts$texts))

embed_xl <- textEmbed(texts_8, model="xlnet-base-cased")

dist_xl <- textDistanceMatrix((embed_xl$texts$texts))

# Packages
library(ggplot2)
library(gridExtra)
library(reshape2) # for grid.arrange

dist_bert
dist_roberta
dist_xl

all_vals <- c(as.vector(dist_bert),
              as.vector(dist_roberta),
              as.vector(dist_xl))

global_min <- min(all_vals, na.rm = TRUE)
global_max <- max(all_vals, na.rm = TRUE)

varnames <- c(
  "Difficult to relax",
  "Irritable",
  "Worried",
  "Nervous/Anxious",
  "Offline interaction (min)",
  "Social media (min)",
  "Time outside (min)",
  "Time at home (min)"
)

rownames(dist_bert) <- varnames
colnames(dist_bert) <- varnames

rownames(dist_roberta) <- varnames
colnames(dist_roberta) <- varnames

rownames(dist_xl) <- varnames
colnames(dist_xl) <- varnames




# Convert a matrix to a ggplot heatmap
make_heatmap <- function(mat, title, palette = "Blues",
                         global_min, global_max) {
  df <- reshape2::melt(as.matrix(mat))
  colnames(df) <- c("Question_X", "Question_Y", "Distance")
  # Force ggplot to use custom variable order
  df$Question_X <- factor(df$Question_X, levels = rownames(mat))
  df$Question_Y <- factor(df$Question_Y, levels = rownames(mat))
  ggplot(df, aes(x = Question_X , y = Question_Y, fill = Distance)) +
    geom_tile() +
    scale_fill_distiller(palette = palette, 
                         direction = 1,
                         limits = c(global_min, global_max)) +
    labs(title = title, fill = "distance") +
    theme_minimal(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
}



p1 <- make_heatmap(dist_Bert, "BERT Distance Matrix",
                   palette = "Blues",
                   global_min, global_max)

p2 <- make_heatmap(dist_roberta, "RoBERTa Distance Matrix",
                   palette = "Blues",
                   global_min, global_max)

p3 <- make_heatmap(dist_xl, "XLNet Distance Matrix",
                   palette = "Blues",
                   global_min, global_max)

gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

png("embedding_heatmaps.png", width = 2400, height = 900, res = 200)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()
