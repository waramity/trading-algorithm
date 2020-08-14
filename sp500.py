import bs4 as bs
import pickle
import requests

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    print(resp)
    print(dir(resp))
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        # short name company
        ticker = row.findAll('td')[0].text
        print(ticker)
        ticker = ticker.rstrip()
        tickers.append(ticker)
    with open('sp500-tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    
    print(tickers)

    return tickers

save_sp500_tickers()
