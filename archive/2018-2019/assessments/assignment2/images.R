library(tidyverse)
ggplot() +
    xlim(0:1) + ylim(0:1) +
    geom_rect(aes(xmin=0.5, xmax=0.5, ymin=0, ymax=1), colour="black", fill=NA) +
    geom_rect(aes(xmin=0, xmax=0.5, ymin=0.75, ymax=0.75), colour="black", fill=NA) +
    geom_rect(aes(xmin=0.25, xmax=0.25, ymin=0, ymax=0.75), colour="black", fill=NA) +
    geom_rect(aes(xmin=0.5, xmax=1, ymin=0.5, ymax=0.5), colour="black", fill=NA) +
    geom_text(data=tribble(
        ~x,  ~y,      ~label,
        0.125, 0.375, "A",
        0.25,  0.875, "B",
        0.375, 0.375, "C",
        0.75,  0.75,  "D",
        0.75,  0.25,  "E"
    ), mapping=aes(x, y, label=label), size=12) +
    theme_bw() + 
    scale_x_continuous(expression(X[1]), expand=c(0,0)) +
    scale_y_continuous(expression(X[2]), expand=c(0,0)) +
    theme(axis.title.y = element_text(angle=0, vjust=0.5)) +
    ggtitle("Figure 1")


ggplot() +
    xlim(0:1) + ylim(0:1) +
    geom_rect(aes(xmin=0.5, xmax=0.5, ymin=0.25, ymax=0.75), colour="black", fill=NA) +
    geom_rect(aes(xmin=0, xmax=0.5, ymin=0.75, ymax=0.75), colour="black", fill=NA) +
    geom_rect(aes(xmin=0, xmax=0.5, ymin=0.25, ymax=0.25), colour="black", fill=NA) +
    geom_rect(aes(xmin=0.25, xmax=0.25, ymin=0.75, ymax=1), colour="black", fill=NA) +
    geom_rect(aes(xmin=0.5, xmax=1, ymin=0.5, ymax=0.5), colour="black", fill=NA) +
    geom_rect(aes(xmin=0.75, xmax=0.75, ymin=0, ymax=0.5), colour="black", fill=NA) +
    geom_text(data=tribble(
        ~x,  ~y,      ~label,
        0.125, 0.875, "A",
        0.25,  0.5, "B",
        0.375, 0.125, "C",
        0.75,  0.75,  "D",
        0.875,  0.25,  "E"
    ), mapping=aes(x, y, label=label), size=12) +
    theme_bw() + 
    scale_x_continuous(expression(X[1]), expand=c(0,0)) +
    scale_y_continuous(expression(X[2]), expand=c(0,0)) +
    theme(axis.title.y = element_text(angle=0, vjust=0.5)) +
    ggtitle("Figure 2")
