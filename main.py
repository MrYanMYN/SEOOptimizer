import os
import dotenv
from datetime import date
from crew import define_pipeline
from trends import fetch_todays_trends


# Get your crew to work!
def write_to_file(trend, result):
    f = open(f'results/{trend}-{date.today()}.txt', 'w')
    f.write(result)
    f.close()

def write_trends_articles(trends):
    for key, item in trends.items():
        create_post(item)


def create_post(item):
    trend, relatedTopics = item['trend'], item['relatedTopics']
    print(f"Item: {trend}")
    crew = define_pipeline(trend, relatedTopics, local=True)
    result = crew.kickoff()
    write_to_file(trend, result)


def main():
    dotenv.load_dotenv()
    trends = fetch_todays_trends("canada")
    write_trends_articles(trends)


if __name__ == "__main__":
    main()
