rm(list = ls())
libs = c('psych')
sapply(libs, require, character.only = TRUE)
rm(libs)

load('~/git/projgamma/ad/bank/bank_raw.Rdata')
# raw = read.csv('~/git/projgamma/ad/bank/bank-additional-full_normalised.csv')

raw$Employment = ''
raw$Employment[which(raw$job.admin. == 1)] = 'Admin'
raw$Employment[which(raw$job.blue.collar == 1)] = 'BlueCollar'
raw$Employment[which(raw$job.entrepreneur == 1)] = 'Entrepreneur'
raw$Employment[which(raw$job.housemaid == 1)] = 'Housemaid'
raw$Employment[which(raw$job.services == 1)] = 'Services'
raw$Employment[which(raw$job.technician == 1)] = 'Technician'
raw$Employment[which(raw$job.retired == 1)] = 'Retired'
raw$Employment[which(raw$job.management == 1)] = 'Management'
raw$Employment[which(raw$job.unemployed == 1)] = 'Unemployed'
raw$Employment[which(raw$job.self.employed == 1)] = 'SelfEmployed'
raw$Employment[which(raw$job.unknown == 1)] = 'Unknown'
raw$Employment[which(raw$job.student == 1)] = 'Student'

raw$MaritalStatus = ''
raw$MaritalStatus[which(raw$marital.single == 1)] = 'Single'
raw$MaritalStatus[which(raw$marital.married == 1)] = 'Married'
raw$MaritalStatus[which(raw$marital.divorced == 1)] = 'Divorced'
raw$MaritalStatus[which(raw$marital.unknown == 1)] = 'Unknown'

raw$EducationStatus = ''
raw$EducationStatus[which(raw$education.illiterate == 1)] = 'Illiterate'
raw$EducationStatus[which(raw$education.basic.4y == 1)] = 'Y4'
raw$EducationStatus[which(raw$education.basic.6y == 1)] = 'Y6'
raw$EducationStatus[which(raw$education.basic.9y == 1)] = 'Y9'
raw$EducationStatus[which(raw$education.high.school == 1)] = 'High School'
raw$EducationStatus[which(raw$education.professional.course == 1)] = 'Professional'
raw$EducationStatus[which(raw$education.university.degree == 1)] = 'University'
raw$EducationStatus[which(raw$education.unknown == 1)] = 'Unknown'

raw$XMonth = ''
raw$XMonth[which(raw$month.mar == 1)] = 'March'
raw$XMonth[which(raw$month.apr == 1)] = 'April'
raw$XMonth[which(raw$month.may == 1)] = 'May'
raw$XMonth[which(raw$month.jun == 1)] = 'June'
raw$XMonth[which(raw$month.jul == 1)] = 'July'
raw$XMonth[which(raw$month.aug == 1)] = 'August'
raw$XMonth[which(raw$month.sep == 1)] = 'September'
raw$XMonth[which(raw$month.oct == 1)] = 'October'
raw$XMonth[which(raw$month.nov == 1)] = 'November'
raw$XMonth[which(raw$month.dec == 1)] = 'December'
raw$Month = factor(
  raw$XMonth, 
  levels = c('March','April','May','June','July','August',
             'September','October','November','December')
  )

raw$XDay = ''
raw$XDay[which(raw$day_of_week.mon == 1)] = 'Monday'
raw$XDay[which(raw$day_of_week.tue == 1)] = 'Tuesday'
raw$XDay[which(raw$day_of_week.wed == 1)] = 'Wednesday'
raw$XDay[which(raw$day_of_week.thu == 1)] = 'Thursday'
raw$XDay[which(raw$day_of_week.fri == 1)] = 'Friday'
raw$Day = factor(
  raw$XDay, 
  levels = c('Monday','Tuesday','Wednesday','Thursday','Friday')
  )

raw$HousingStatus = ''
raw$HousingStatus[which(raw$housing.1 == 1)] = '1'
raw$HousingStatus[which(raw$housing.0 == 1)] = '0'
raw$HousingStatus[which(raw$housing.unknown == 1)] = 'Unknown'

raw$DefaultStatus = ''
raw$DefaultStatus[which(raw$default.0 == 1)] = '0'
raw$DefaultStatus[which(raw$default.1 == 1)] = '1'
raw$DefaultStatus[which(raw$default.unknown == 1)] = 'Unknown'

raw$LoanStatus = ''
raw$LoanStatus[which(raw$loan.0 == 1)] = '0'
raw$LoanStatus[which(raw$loan.1 == 1)] = '1'
raw$LoanStatus[which(raw$loan.unknown == 1)] = 'Unknown'

raw$POutcomeStatus = ''
raw$POutcomeStatus[which(raw$poutcome.nonexistent == 1)] = 'Nonexistent'
raw$POutcomeStatus[which(raw$poutcome.failure == 1)] = 'Failure'
raw$POutcomeStatus[which(raw$poutcome.success == 1)] = 'Success'

raw$PDays = !(raw$pdays < 1)
raw$Previous = raw$previous > 0

raw$EmpVarRate = cut(raw$emp.var.rate, c(0, 0.2, 0.5, 0.7, 1.), include.lowest = TRUE)
levels(raw$EmpVarRate) = c('Low','Moderate','Medium','High')
raw$EmpVarRate = as.character(raw$EmpVarRate)

raw$Age = cut(raw$age, c(0, 0.1, 0.55, 1), include.lowest = TRUE)
levels(raw$Age) = c('Low','Medium','High')
raw$Age = as.character(raw$Age)

raw$Duration = cut(raw$duration, c(0, 0.1, 1), include.lowest = TRUE)
levels(raw$Duration) = c('Short','Long')
raw$Duration = as.character(raw$Duration)

raw$NEmployed = cut(raw$nr.employed, c(0, 0.5, 1), include.lowest = TRUE)
levels(raw$NEmployed) = c('Low','High')
raw$NEmployed = as.character(raw$NEmployed)

jobfields = c('Admin','BlueCollar','Entrepreneur','Housemaid',
              'Management','SelfEmployed','Services','Technician')
raw$Employment[which(raw$Employment %in% jobfields)] = 'Employed'

raw$EducationStatus[raw$EducationStatus %in% c('Illiterate','Y4','Y6','Y9')] = 'BelowHigh'
raw$EducationStatus[raw$EducationStatus %in% c('Professional','University')] = 'Degree'

raw$DefaultStatus[which(raw$DefaultStatus %in% c('0','1'))] = 'Known'

cols = c("Age", 'Employment', 'MaritalStatus', 'EducationStatus', 'DefaultStatus',
         'HousingStatus', 'LoanStatus', 'Duration', 'PDays',
         'Previous', 'POutcomeStatus', "EmpVarRate", "NEmployed", "class")
df = raw[cols]

# months = c('March','April','May','June','July','August','September','October',
#           'November','December')
# t = table(factor(df$XMonth, levels = months), df$class)
# print(t)
# chisq.test(t) # Month is strongly associated with anomaly

# days = c('Monday','Tuesday','Wednesday','Thursday','Friday')
# t = table(factor(df$Xday, levels = days), df$class)
# print(t)
# chisq.test(t) # day of week is weakly associated with anomaly

# t = table(df$XMonth, df$Xday, df$class)
# t2 = array(dim = dim(t))
# t2[,,1] = table(df$XMonth, df$Xday); t2[,,2] = table(df$XMonth, df$Xday)
# print(t / t2)
# mantelhaen.test(t) # finds significant, but looking at the table, I don't find
#                    # a strong effect from the day of week association.

# t = table(df$Employment, df$class)
# chisq.test(t) # strongly significant
# print(t / rowSums(t)) # seems to indicate that retired and students are main
#                       # contributors to the anomaly count


# t = table(df$MaritalStatus, df$class)
# print(t / rowSums(t))
# chisq.test(t) # significant, but not strongly predictive.
#               # No further reduction available.

# t = table(df$EducationStatus, df$class)
# print(t / rowSums(t))
# chisq.test(t) # significant.  Y4-9 seem to act similar.  So do Degreed.
#               # group illiterate with below high school, as there's only 18 of them.
# df$EducationStatus[df$EducationStatus %in% c('Illiterate','Y4','Y6','Y9')] = 'BelowHigh'
# df$EducationStatus[df$EducationStatus %in% c('Professional','University')] = 'Degree'

# t = table(df$LoanStatus, df$class) # not significant?
# chisq.test(t)
# t = table(df$DefaultStatus == 'Unknown', df$class)
# chisq.test(t)
# t = table(df$LoanStatus, df$DefaultStatus, df$class)
# t2 = array(dim = dim(t))
# t2[,,1] = table(df$LoanStatus, df$DefaultStatus)
# t2[,,2] = table(df$LoanStatus, df$DefaultStatus)
# print(t / t2)
# mantelhaen.test(t)  # No association
# df$DefaultStatus[which(df$DefaultStatus %in% c('0','1'))] = 'Known'

# table(raw$emp.var.rate, raw$class)
# table(raw$emp.var.rate, raw$class) / rowSums(table(raw$emp.var.rate, raw$class))
# chisq.test(table(raw$emp.var.rate, raw$class))

# raw$agecat = cut(raw$age, seq(0, 1, by = 0.05), include.lowest = TRUE)
# table(raw$agecat, raw$class) / rowSums(table(raw$agecat, raw$class))
# lowest category suspicious, categories above 0.5 suspicious.

keep_rows = sample(nrow(df), size = 5000, replace = FALSE)
write.csv(df[names(df)[1:13]][keep_rows,], file = './cat_data.csv', row.names = FALSE)
write.csv(df[c('class')][keep_rows,], file = './cat_outcome.csv', row.names = FALSE)

rows_0 = df[df$class == 0,]
rows_1 = df[df$class == 1,]

for(i in 1:10){
  nrow_0 = round(5000 * (1 - (i * 0.01)))
  nrow_1 = 5000 - nrow_0
  keep_rows_0 = sample(nrow(rows_0), size = nrow_0, replace = FALSE)
  keep_rows_1 = sample(nrow(rows_1), size = nrow_1, replace = FALSE)
  tdf = rbind(rows_0[keep_rows_0,], rows_1[keep_rows_1,])
  write.csv(
    tdf[names(tdf)[1:13]], 
    file = paste0('./bank_',i,'/cat_data.csv'), 
    row.names = FALSE
    )
  write.csv(
    tdf[c('class')], 
    file = paste0('./bank_',i,'/cat_outcome.csv'), 
    row.names = FALSE
    )
}





