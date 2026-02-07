import csv
import random
import re

# Define diverse datasets for generating realistic queries
artists = [
    "Ella Fitzgerald", "Duke Ellington", "Crystal Gayle", "Ricchi e Povery", "Taylor Swift", "Drake", "Eminem", 
    "Ed Sheeran", "Adele", "Bruno Mars", "Rihanna", "Coldplay", "Imagine Dragons", "Daft Punk",
    "Miles Davis", "John Coltrane", "Louis Armstrong", "Ella Fitzgerald", "Frank Sinatra", 
    "Bob Marley", "Jimi Hendrix", "Led Zeppelin", "Pink Floyd", "Nirvana", "Radiohead",
    "Billie Holiday", "Duke Ellington", "Charlie Parker", "Thelonious Monk", "John Legend",
    "Kendrick Lamar", "Travis Scott", "Billie Eilish", "Post Malone", "The Weeknd", "Dua Lipa",
    "BTS", "Bad Bunny", "Shakira", "Rosalía", "Hans Zimmer", "John Williams", "Ludwig van Beethoven",
    "Wolfgang Amadeus Mozart", "Johann Sebastian Bach", "Frederic Chopin", "Pyotr Ilyich Tchaikovsky"
]

titles = [
    "Let's Do It", "Cotton tail", "Shape of You", "Acapulco", "Uptown Funk",
    "Blinding Lights", "Dance Monkey", "Someone Like You", "Halo", "Bad Guy", "Old Town Road",
    "Despacito", "Happy", "Shallow", "All of Me", "Thinking Out Loud", "Watermelon Sugar",
    "Levitating", "Piano Man", "Hotel California", "Stairway to Heaven", "Smells Like Teen Spirit",
    "Take Five", "So What", "Take the A Train", "My Favorite Things", "Feeling Good",
    "Fly Me to the Moon", "What a Wonderful World", "Imagine", "Yesterday", "Hey Jude",
    "Purple Haze", "Sweet Child O' Mine", "Bohemian Rhapsody", "Another Brick in the Wall",
    "Moon River", "Summertime", "Autumn Leaves", "Blue in Green", "Take Five", "Birdland",
    "Clair de Lune", "Für Elise", "Symphony No. 5", "Ride of the Valkyries", "Nocturne Op. 9 No. 2"
]

genres = [
    "jazz", "rock", "pop", "hip hop", "classical", "electronic", "blues", "country", "reggae",
    "r&b", "soul", "funk", "disco", "metal", "punk", "folk", "indie", "ambient", "techno",
    "house", "trance", "bossa nova", "salsa", "flamenco", "opera", "symphony", "chamber music",
    "bebop", "cool jazz", "smooth jazz", "fusion", "trip hop", "lo-fi", "dubstep", "drum and bass"
]

albums = [
    "Stockholm Concert", "Greatest Hits", "Italy", "Back in Black", "Rumours", "21",
    "÷ (Divide)", "Lemonade", "Good Girl Gone Bad", "25", "After Hours", "Future Nostalgia",
    "Scorpion", "Views", "To Pimp a Butterfly", "When We All Fall Asleep, Where Do We Go?",
    "Kind of Blue", "Time Out", "Ellington at Newport", "A Love Supreme", "Giant Steps",
    "Round About Midnight", "The Joshua Tree", "Nevermind", "OK Computer", "Back to Black",
    "1989", "Reputation", "Lover", "Folklore", "Evermore", "Random Access Memories",
    "Random Access Memories", "Random Access Memories", "The Blueprint", "My Beautiful Dark Twisted Fantasy",
    "Moonlight Sonata", "The Four Seasons", "Eine kleine Nachtmusik", "The Magic Flute"
]

# Verbs and phrasings for natural language variation
play_verbs = [
    "play", "start", "begin", "put on", "queue up", "launch", "initiate", "run", "stream",
    "let me hear", "I want to listen to", "give me", "bring up", "pull up", "load",
    "spin up", "fire up", "turn on", "get me", "can you play", "please play", "would you play"
]

# Generate 400 music-related queries with diverse field combinations
music_queries = []

# Helper function to generate XML response
def generate_response(artist="", title="", genre="", album=""):
    return f"<tool_call><name>audioplay.play_request</name><arguments><artist>{artist}</artist><title>{title}</title><genre>{genre}</genre><album>{album}</album></arguments></tool_call>"

# 1. Artist only (60 examples)
for i in range(60):
    artist = random.choice(artists)
    verb = random.choice(play_verbs)
    # Vary phrasing
    phrasing_type = random.randint(0, 3)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} {artist}"
    elif phrasing_type == 1:
        request = f"{verb.capitalize()} music by {artist}"
    elif phrasing_type == 2:
        request = f"I want to hear {artist}"
    else:
        request = f"Play something from {artist}"
    music_queries.append((request, generate_response(artist=artist)))

# 2. Title only (60 examples)
for i in range(60):
    title = random.choice(titles)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 3)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} '{title}'"
    elif phrasing_type == 1:
        request = f"Play the song called {title}"
    elif phrasing_type == 2:
        request = f"I'd like to hear {title}"
    else:
        request = f"Start playing {title}"
    music_queries.append((request, generate_response(title=title)))

# 3. Genre only (60 examples)
for i in range(60):
    genre = random.choice(genres)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 4)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} some {genre}"
    elif phrasing_type == 1:
        request = f"Play {genre} music"
    elif phrasing_type == 2:
        request = f"I'm in the mood for {genre}"
    elif phrasing_type == 3:
        request = f"Give me some good {genre}"
    else:
        request = f"Queue up a {genre} playlist"
    music_queries.append((request, generate_response(genre=genre)))

# 4. Album only (50 examples)
for i in range(50):
    album = random.choice(albums)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 3)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} the album '{album}'"
    elif phrasing_type == 1:
        request = f"Play {album} album"
    elif phrasing_type == 2:
        request = f"Start the album {album}"
    else:
        request = f"I want to listen to {album}"
    music_queries.append((request, generate_response(album=album)))

# 5. Artist + Title (50 examples)
for i in range(50):
    artist = random.choice(artists)
    title = random.choice(titles)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 3)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} '{title}' by {artist}"
    elif phrasing_type == 1:
        request = f"Play {artist}'s {title}"
    elif phrasing_type == 2:
        request = f"I want to hear {title} from {artist}"
    else:
        request = f"Start {title} performed by {artist}"
    music_queries.append((request, generate_response(artist=artist, title=title)))

# 6. Artist + Genre (30 examples)
for i in range(30):
    artist = random.choice(artists)
    genre = random.choice(genres)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 2)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} {genre} music by {artist}"
    elif phrasing_type == 1:
        request = f"Play {artist} in the {genre} style"
    else:
        request = f"I want to hear {artist} doing {genre}"
    music_queries.append((request, generate_response(artist=artist, genre=genre)))

# 7. Genre + Album (30 examples)
for i in range(30):
    genre = random.choice(genres)
    album = random.choice(albums)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 2)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} the {genre} album '{album}'"
    elif phrasing_type == 1:
        request = f"Play {album} which is {genre}"
    else:
        request = f"Start the {genre} record {album}"
    music_queries.append((request, generate_response(genre=genre, album=album)))

# 8. Title + Album (25 examples)
for i in range(25):
    title = random.choice(titles)
    album = random.choice(albums)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 2)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} '{title}' from the album '{album}'"
    elif phrasing_type == 1:
        request = f"Play {title} off {album}"
    else:
        request = f"Start the track {title} on {album}"
    music_queries.append((request, generate_response(title=title, album=album)))

# 9. Artist + Album (20 examples)
for i in range(20):
    artist = random.choice(artists)
    album = random.choice(albums)
    verb = random.choice(play_verbs)
    phrasing_type = random.randint(0, 2)
    if phrasing_type == 0:
        request = f"{verb.capitalize()} {artist}'s album '{album}'"
    elif phrasing_type == 1:
        request = f"Play the album {album} by {artist}"
    else:
        request = f"Start {album} from {artist}"
    music_queries.append((request, generate_response(artist=artist, album=album)))

# 10. All fields combined (15 examples)
for i in range(15):
    artist = random.choice(artists)
    title = random.choice(titles)
    genre = random.choice(genres)
    album = random.choice(albums)
    request = f"Play '{title}' by {artist} from the album '{album}' in {genre} style"
    music_queries.append((request, generate_response(artist=artist, title=title, genre=genre, album=album)))

# Ensure we have exactly 400 music queries (in case of rounding)
music_queries = music_queries[:400]

# Generate 100 non-music queries
non_music_queries = []

# Weather queries
weather_locations = ["Moscow", "London", "New York", "Tokyo", "Paris", "Berlin", "Sydney", "Rio de Janeiro", "Cairo", "Mumbai"]
for loc in weather_locations:
    non_music_queries.append((f"What's the weather in {loc}?", f"What's the weather in {loc}?"))
    non_music_queries.append((f"How's the weather today in {loc}?", f"How's the weather today in {loc}?"))
    non_music_queries.append((f"Will it rain in {loc} tomorrow?", f"Will it rain in {loc} tomorrow?"))

# Time queries
time_queries = [
    "What time is it?", "What's the current time?", "Tell me the time", "What time is it in Tokyo?",
    "What's the time in London right now?", "How late is it?", "Is it midnight yet?",
    "What time does the sun rise?", "When does the sun set today?"
]
for q in time_queries:
    non_music_queries.append((q, q))

# General knowledge
general_queries = [
    "Who is the president of the United States?", "How tall is Mount Everest?", 
    "What is the capital of France?", "When was Albert Einstein born?", 
    "How many planets are in our solar system?", "What is the speed of light?",
    "Who wrote Romeo and Juliet?", "What is the largest ocean on Earth?",
    "How do airplanes fly?", "Why is the sky blue?", "What is photosynthesis?",
    "How many bones are in the human body?", "What is the square root of 144?",
    "Calculate 45 times 67", "What is 128 divided by 16?", "Solve this equation: 2x + 5 = 15",
    "Translate 'hello' to Spanish", "How do you say 'thank you' in Japanese?",
    "What does 'carpe diem' mean?", "Tell me a joke", "Make me laugh",
    "What's the meaning of life?", "Are we alone in the universe?",
    "What's your favorite color?", "Do you dream?", "Can you feel emotions?",
    "What day is it today?", "What's the date tomorrow?", "How many days until Christmas?",
    "Recommend a good book", "Suggest a movie to watch", "What should I cook for dinner?",
    "How do I tie a tie?", "Explain quantum physics simply", "What is blockchain?",
    "How to learn Python programming?", "Best exercises for weight loss",
    "How to meditate for beginners", "What's happening in the news today?",
    "Summarize today's headlines", "Who won the last World Cup?",
    "What's the score of the Lakers game?", "When is the next SpaceX launch?"
]
for q in general_queries[:40]:
    non_music_queries.append((q, q))

# Ensure we have exactly 100 non-music queries
non_music_queries = non_music_queries[:100]

# Combine all queries
all_queries = music_queries + non_music_queries

# Verify counts
print(f"Music queries: {len(music_queries)}")
print(f"Non-music queries: {len(non_music_queries)}")
print(f"Total queries: {len(all_queries)}")

# Write to CSV
with open('audio_assistant_queries.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['Request', 'Response'])
    for req, resp in all_queries:
        writer.writerow([req, resp])

print("CSV file created successfully")

# Show sample of the data
print("\nSample of generated data:")
for i in range(5):
    print(f"{i+1}. Request: {all_queries[i][0]}")
    print(f"   Response: {all_queries[i][1]}")
print("\nNon-music sample:")
for i in range(400, 405):
    print(f"{i+1}. Request: {all_queries[i][0]}")
    print(f"   Response: {all_queries[i][1]}")

# Result 
