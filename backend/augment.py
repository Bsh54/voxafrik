"""Generate and translate synthetic examples so the model also handles clean, formal input."""
import csv
from translate import translate

LANGUAGES = ["sw", "ha", "ig", "rw", "yo"]

SENTENCES = {
    0: [
        "Good morning everyone have a blessed day", "Thank you so much for helping me yesterday",
        "The match last night was really exciting", "Please send me the notes when you can",
        "I love weekends with my family", "Weather is lovely today lets go out",
        "Congrats on the new job you earned it", "Lets work together for our community",
        "Market was busy but prices were fair", "Our team finished the work early",
        "Welcome to the city we are glad", "Remember to drink water and rest",
        "She cooked a wonderful meal for us", "The children are playing in the park",
        "God bless our hardworking farmers", "Happy birthday my dear friend",
        "The bus was late but we made it", "Lets plan the wedding together",
        "Great goal by our striker tonight", "Studying hard for my exams this week",
        "The new hospital will help many people", "I am grateful for my supportive neighbors",
        "Our church choir sang beautifully today", "Please join us for the community cleanup",
        "The harvest was good this season", "Lets support local businesses this month",
        "Wishing you a safe journey home", "The teacher explained the lesson clearly",
        "We planted trees along the main road", "Our youth group organized a football match",
    ],
    1: [
        "You are so stupid you never get it", "Shut up nobody wants your nonsense",
        "You are a complete idiot and fool", "Get lost you are useless and annoying",
        "You are pathetic and weak grow up", "Stop talking rubbish you clown",
        "You are dumb your ideas are garbage", "Nobody likes you you are a loser",
        "You are a liar and a coward", "Your work is trash you are useless",
        "You disgust me you selfish fool", "Keep your stupid mouth shut",
        "You are an embarrassment to everyone", "What a worthless lazy person you are",
        "You always mess everything up idiot", "Only a moron would say that",
        "You are so annoying and foolish", "Go away you brainless fool",
        "Such a pathetic excuse of a person", "You never do anything right stupid",
        "Everyone is tired of your stupid attitude", "You are a shameless clumsy fool",
        "Why are you so dumb and slow", "Your opinion is worthless shut it",
        "You are a pathetic little coward", "Stop embarrassing yourself you idiot",
        "Nobody respects a liar like you", "You are weak and completely useless",
        "Get a brain you absolute fool", "You talk too much and know nothing",
    ],
    2: [
        "People like you should not live here", "Your tribe is a disease leave our country",
        "We must get rid of all these foreigners", "Those people are animals with no rights",
        "Your kind is worthless and should vanish", "We should drive them all out of our land",
        "People of that religion are evil remove them", "They have no place among us they must go",
        "Your ethnic group should be banished from here", "No one from your community belongs here",
        "Those outsiders are a burden we must remove", "People like them should be banished forever",
        "We do not want your people in our town", "That group should be wiped from this place",
        "All of them should be chased away now", "Your people are a plague on this nation",
        "Send them back they are not welcome", "We will never accept people like you here",
        "That community does not deserve to exist", "They are subhuman and must be expelled",
        "Those settlers must be forced out today", "Your race has no value and should disappear",
        "We should ban that tribe from the market", "People of that faith should be silenced forever",
        "Nobody of your kind should hold any job", "That minority must be removed from our schools",
        "We must cleanse our region of those people", "Their whole village should be driven away",
        "Those immigrants are vermin get them out", "People like you poison our beautiful country",
    ],
}


def build(output_path):
    rows = []
    for label, sentences in SENTENCES.items():
        for sentence in sentences:
            for language in LANGUAGES:
                try:
                    translated = translate(sentence, language)
                except Exception:
                    translated = ""
                if translated.strip():
                    rows.append((translated.strip(), label))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print("saved", build("augment.csv"), "rows")
