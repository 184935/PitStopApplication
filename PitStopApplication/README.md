A web application to attempt predictions of pitstops. You enter the Tyre age, Compound, Year, Lap number, Total laps, position, stint, pitstop, and position change. 
The model will then return whether it thinks the car will pit on the next lap, and the probability of it pitting. The threshold has been set to 0.3, in order to catch more actual pitstops without too big drawbacks.

The model is trained and tested in ML.NET, while the data is visialized and adjusted with python. The web application runs in ASP.NET Razor pages. 

The website can be reached here: https://pitstopprediction-euaaezfkbng2eyd4.swedencentral-01.azurewebsites.net/
