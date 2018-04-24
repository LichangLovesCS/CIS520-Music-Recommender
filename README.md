# CIS520-Music-Recommender
Final Project for CIS520 (Machine Learning) at Upenn

# Running the Web Interface 
To run the web interface, download the last.fm dataset from http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html, unzip the datasets into the same directory as the source code. 

Then run python3 prepare.py (make sure pandas and relevant modules are installed properly. This can be done easily with tools like HomeBrew or Pip). This will take several minutes as the dataset is large and will output a temporary csv file with clean data. 

Finally run python3 s.py and open the browser and type localhost:5000. The web interface should then start running.
