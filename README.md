# Api_Sentiments

git clone https://github.com/SIRSTN/Api_Sentiments.git
cd api_sentiments
git add .
git commit -m "cloud commit"
git push -u origin main

git fetch origin
git merge origin/main
git pull origin main

git reset --hard

node server.js
http://localhost:3000/api/google-trends-realtime/US/b