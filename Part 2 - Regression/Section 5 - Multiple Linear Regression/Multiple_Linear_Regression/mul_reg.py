#dataset 
# anonymised dataset of start ups
# administrative costs = paying employees, etc

# goal: venture capitalist has hired me to create a model that 
# predicts or tells which start up is most promising and should be
# invested in

# profit is dependent variable

# multiple linear regression - more than one independent variable
# Assumptions of a linear regression:
# 1. Linearity
# 2. Homoscedasticity
# 3. Multivariate normality 
# 4. Independence of errors
# 5. Lack of multicollinearity

# before building a linear regression model you should if these
# assumptions are true. This is more of cautionary note, while 
# modelling a linear regression model in real life. It won't be dealt
# with in this course. 

# Dummy Variables for state column
# State is a categorical variable and hence, this requires pre processing
# phenomenon of dummy variable trap 
# dummyVar1 = 1 - dummyVar2, this equation implies dummyVar1 is dependent on 
# dummyVar2 and hence what happens is that the model cannot distinguish
# the effects of dummyVar1 from dummyVar2. 
