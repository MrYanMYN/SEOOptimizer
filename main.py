from crew import define_blog_topic
from trends import fetch_todays_trends
# Get your crew to work!

trends = fetch_todays_trends("united_states")
for key, result in trends.items():
  crew = define_blog_topic(result['trend'], result['relatedTopics'])
  result = crew.kickoff()