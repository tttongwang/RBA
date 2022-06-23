
library(ggplot2)
library(gridExtra)
library(grid)

theme_update(panel.grid.major=element_blank(),
             panel.grid.minor=element_blank(),
             panel.background=element_blank(),
             axis.line=element_line(colour='black'),
             plot.title = element_text(hjust = 0.5,size=7,face = "bold"),
             legend.text = element_text(size=7,face = "bold"),
             axis.title=element_text(size=7,face = "bold"),
             axis.text=element_text(size=7,face = "bold")
)

update_geom_defaults("line", list(size=0.7))

colors <- c("contextual_bandits" = "grey25",
            "RBA" = "cornflowerblue",
            "suggested_W"="firebrick3", 
            "controlled_std_W"="goldenrod1")



grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
  
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position="none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
  
  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))
  
  grid.newpage()
  grid.draw(combined)
  
  # return gtable invisibly
  invisible(combined)
  
}

