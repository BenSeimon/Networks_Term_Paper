#source: https://cran.r-project.org/web/packages/matconv/vignettes/overallUse.html#:~:text=If%20you%20just%20want%20a,with%20just%20the%20Matlab%20code.&text=The%20function%20outputs%20a%20list,exactly%20the%20program%20is%20doing.

library('matconv')
library('tidyverse')
load_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Mexican Drug War Data/AER2012-1637data/'
save_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Melissa Dell R Code/'
setwd(load_path)

mask <- str_detect(list.files(), "\\.m$")
matlab_files <- list.files()[mask]


for (file in matlab_files){
  mat2r(inMat = paste0(load_path, file), pathOutR = paste0(save_path, substr(file ,start = 1, stop = nchar(file)-2), '.R'))
}


