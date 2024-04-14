import requests
import json
def call_news():

    url = "https://google-news-api1.p.rapidapi.com/search"

    querystring = {"language":"EN","limit":50}

    '''headers = {
	"X-RapidAPI-Key": "52af1040b7msh041a60921f67577p1a1eabjsn10a72f629937-----sss",
	"X-RapidAPI-Host": "google-news-api1.p.rapidapi.com"
}

    response = requests.get(url, headers=headers, params=querystring)

    #print(response.json())
    with open("resp_json.json", "w") as outfile: 
        json.dump(response.json(), outfile)'''
    f = open('resp_json.json')
 
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    #print(data)
    all_news = data["news"]["news"]
    #print(all_news)
    all_news_desc=[]
    all_news_links=[]
    all_news_tups = []
    for news in all_news:
        if news["title"] is None or news["description"] is None : 
            continue
        all_news_desc.append(news["title"]+news["description"])
        all_news_links.append(news["link"])
        all_news_tups.append((news["title"],news["description"],news["link"]))
    return all_news_desc,all_news_links,all_news_tups