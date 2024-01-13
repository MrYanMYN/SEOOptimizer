from pytrends.request import TrendReq

def get_related_queries(trend: str):
    pytrend = TrendReq()
    kw_list = [trend]
    pytrend.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
    related_queries = pytrend.related_queries()
    return related_queries

def fetch_todays_trends(region):
    pytrend = TrendReq()
    df = pytrend.trending_searches(pn=region)
    trends = {}
    tmp_df = df

    for i in range(0, 19):
        # trends.append(df.at[i, 0])
        currentTrend = df.at[i, 0]
        while True:
            try:
                topics = pytrend.suggestions(keyword=currentTrend)
                filtered_topics = [{'title': topic['title'], 'type': topic['type']} for topic in topics]
                trends[i] = {"trend": df.at[i, 0], "relatedTopics": filtered_topics}
                break

            except:
                pass

        df = tmp_df

    return trends

# print(get_related_queries("Crypto"))
# # print(fetch_todays_trends('united_states'))