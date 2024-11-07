import requests
from bs4 import BeautifulSoup
from pathlib import Path

def getTranscripts():
	page = requests.get("https://vanisource.org/wiki/Category:Lectures_-_Bhagavad-gita_As_It_Is")
	soup = BeautifulSoup(page.content, "html.parser")

	pages_container = soup.find("div", class_="mw-content-ltr")
	page_lists = pages_container.find_all("ul")
	page_links = []

	for i in range(1, len(page_lists)):
		page_links.extend(page_lists[i].find_all("a"))

	Path.mkdir(Path("transcripts"), exist_ok=True)

	# For each page, save the transcript to a txt file
	for link in page_links:
		print(link["title"])
		# Open lecturex
		page = requests.get("https://vanisource.org" + link["href"])
		soup = BeautifulSoup(page.content, "html.parser")

		text_container = soup.find("div", id="bodyContent")
		for text in text_container.find("div", class_="mw-parser-output").findChildren(recursive=False):
			if text.name == "p" and text.text != "\n":

				with open("transcripts/" + link["title"] + ".txt", "a", encoding='utf-8') as f:
					f.write(text.text + "\n")

			elif text.name == "dl":
				with open("transcripts/" + link["title"] + ".txt", "a", encoding='utf-8') as f:
					f.write(text.text + "\n\n")

getTranscripts()