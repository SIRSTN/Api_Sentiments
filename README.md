# Api_Sentiments

git clone https://github.com/SIRSTN/Api_Sentiments.git
git clone https://github.com/SIRSTN/Connector_Mastodon.git
git clone https://github.com/SIRSTN/Connector_Reddit.git
git clone https://github.com/Philipid3s/google-trends-api

cd api_sentiments
git add .
git commit -m "Sentiment commit"
git push -u origin main

git fetch origin
git merge origin/main
git pull origin main

node server.js
http://localhost:3000/api/google-trends-realtime/US/b