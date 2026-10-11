So lets Decode what Gradient Boosting is another Ensemble Technique 
We all know In Machine learning are main is to find a function of X f(x) which map the best relationship between X and y let take an example if we have a dataset and we plot it and u can see a wave then it means our relationship is kinda sine so we can say y = sin(x) but what if dataset is too complex and we have to find so instead of finding the whole pattern(function) at onces Gradient Boosting builds a complex function step by step by adding small models, where each new model tries to correct the errors made by the previous models. wait what I mean ? See the img what you see a curve and I say find a function for it i mean find a relation between X and y at first it look complex many will say use polynomial regression but due to Runge's phenomenon the error may increase if use higher degree so let get back as you see the graph and we try to find a function f(x) and the function is just sin(x) + x (I modify it by multiplying x\*2 for Visualisation) 
 So if we write it it would be 
y = sin(x) + x
it is a combination of 2 functions 
y = f1(x) + f2(x)-> sin(x) and x 

 We approximate the complex function using a sum of many smaller models. These smaller models don't necessarily represent the actual components each one mainly learns what the current model is still getting wrong and thats all what Gradient Boosting is all about we make smaller functions and then sum them all and that how we get the big function this is called Addictive Modelling adding up smaller models 
Gradient Boosting

In Gradient Boosting we start with a simple thing mean we make a new array of the len of X\_train and fill with mean of y\_train means our model 0 function 0 is nothing but the mean f0(x) = np.mean(y\_train) next step calculate 

Residual(error) actual(y) - pred(prediction array) 

residual = y - prediction then fit  a weak learner commonly small shallow trees the depth can be tune and In tree we put X\_train and Residual we know the pred will not be good then update prediction, prediction = prediction + learning\_rate\*model(our DT) also we append our tree each time for future predictions ok unseen dataset we add learning rate to reduce overfitting and it's complete fit function is done now Predictions it's pretty simple we start with a var called prediction should be len of X\_test and fill with mean of Y and loop through all trees pred X\_test and keep multiply prediction with learning\_rate and append it to our final prediction and then return it 

 Thing to what happened we start with a function 

f0(x) -> Nothing but Mean of Y 

f1(X) -> made by training on X and Residual our error means at first we put our errors as y and prediction and after the prediction update old prediction:

 prediction = prediction + lr\*model(our current tree) 

F2(x) -> same as F1(X) 

F3(X) -> same 

 For an unseen dataset, we start with the initial prediction f0(x) then pass through every tree and add its scaled contribution to the current prediction.

And this was all i put in my brain 🧠 
