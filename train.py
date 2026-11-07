import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

JOY = [
 "I just got the internship I have been dreaming about for months",
 "We won the cricket match in the final over what a feeling",
 "My birthday surprise party was the best day of my life",
 "I finally finished my project and it works perfectly",
 "Dancing with my friends at the college fest felt amazing",
 "I scored the highest marks in my class today",
 "The sunset at the beach with my family was beautiful",
 "I laughed so hard at the comedy show last night",
 "My sister got accepted into her dream university today",
 "I bought my first bike with my own savings",
 "The team celebrated our hackathon win with ice cream",
 "Waking up to good news always makes my whole day",
 "I passed my driving test on the first try",
 "Our college band performed and the crowd cheered loudly",
 "I found my old childhood photos and felt so happy",
 "The festival lights and music filled the whole street",
 "My best friend is visiting me after two long years",
 "I cooked my favourite dish and it turned out delicious",
 "We reached the top of the hill and the view was breathtaking",
 "My dog ran to greet me the moment I came home",
 "I got a scholarship for my final year studies",
 "The movie we watched together made everyone laugh",
 "I completed my first 10 kilometre run this morning",
 "My parents surprised me with a new laptop for college",
 "The rain stopped and a rainbow appeared over the campus",
 "I finally understood the topic that confused me for weeks",
 "My artwork was displayed at the college exhibition today",
 "We had a picnic under the big tree near the lake",
 "I received a thank you message from someone I helped",
 "The concert last night was the most fun I have had all year",
]
SADNESS = [
 "I miss my grandmother who passed away last year",
 "My best friend moved to another country and I feel alone",
 "I failed the exam even after studying so hard",
 "The rain reminds me of the days we spent together",
 "I lost my favourite watch that my father gave me",
 "Nobody came to my birthday party this year",
 "I feel empty since my pet dog died last week",
 "My team lost the match in the last minute again",
 "I have been feeling lonely in the hostel lately",
 "The letter I waited for never arrived in the mail",
 "I miss the food my mother cooks at home",
 "My old phone broke and all my photos are gone",
 "I could not attend my sister's wedding because of exams",
 "The tree in our garden where I played as a child was cut down",
 "I feel sad when I see my old school friends moving on",
 "My favourite book got wet in the rain and ruined",
 "I waited for hours but nobody called me back",
 "The last day of college felt like saying goodbye forever",
 "I lost my wallet with all my important cards inside",
 "It hurts to watch my parents grow old and tired",
 "I could not help my friend when he needed me most",
 "The empty house feels quiet without my family around",
 "My dreams feel far away and out of reach today",
 "I miss the laughter we shared during summer holidays",
 "The song on the radio reminds me of someone I lost",
 "I feel left out when my friends make plans without me",
 "My plant that I grew for months has finally withered",
 "I cried quietly after reading the old messages again",
 "The farewell speech made everyone in the room tear up",
 "I feel homesick every night in this new city",
]
ANGER = [
 "Someone stole my bicycle from outside the college gate",
 "I am furious that the shopkeeper cheated me with a fake product",
 "The bus left without waiting even though I was running",
 "My roommate used my laptop without asking and broke it",
 "I am angry that the teacher blamed me for something I did not do",
 "The internet stopped working right before my online exam",
 "Someone cut in front of me in the long queue again",
 "I hate it when people lie to my face so shamelessly",
 "The mechanic charged me double for a simple repair",
 "My neighbour plays loud music every night till 2 am",
 "I am mad that my hard work was given to someone else",
 "The delivery person threw my parcel over the gate",
 "People keep parking their cars in front of my house",
 "I am annoyed that the meeting was cancelled without any notice",
 "The waiter ignored our table for half an hour",
 "Someone spread false rumours about me in college",
 "I am furious that my train was delayed by five hours",
 "The shop refused to refund my money for a damaged item",
 "My brother ate my share of the sweets again",
 "I am angry that nobody listens to my complaints",
 "The landlord increased the rent without any warning",
 "Someone scratched my new bike in the parking lot",
 "I am irritated that the software crashed and lost my work",
 "The guard was rude to my parents at the gate",
 "People littering the park every day makes me so mad",
 "I am annoyed that my friend shared my secret with everyone",
 "The bank deducted money from my account without reason",
 "Someone took my seat in the library after I left my bag",
 "I am furious that the exam paper had so many errors",
 "The driver splashed muddy water all over my clothes",
]
FEAR = [
 "I heard strange noises in the empty house last night",
 "The thunder and lightning during the storm scared me",
 "I am afraid of failing my final exams this semester",
 "Walking alone on the dark road made my heart race",
 "I worry that my parents might fall sick far away",
 "The lift stopped between floors and I panicked",
 "I am scared of speaking in front of a large audience",
 "The dog barked loudly and chased me down the street",
 "I fear losing my scholarship if my grades drop",
 "The news about the earthquake near our town frightened me",
 "I am anxious about my job interview tomorrow morning",
 "The dark tunnel on the mountain road gave me chills",
 "I worry that I might lose my way in the new city",
 "The creepy old building at the end of the street haunts me",
 "I am terrified of heights when I stand on the balcony",
 "The sudden blackout in the middle of the night scared us",
 "I fear that my best friend is angry with me forever",
 "The snake I saw near the garden made me freeze",
 "I am nervous about the results coming out tomorrow",
 "The storm waves crashing on the beach looked dangerous",
 "I worry about what will happen if I cannot pay the fees",
 "The ghost stories my friends told kept me awake all night",
 "I am afraid of the deep water when I swim in the sea",
 "The strange man following me on the road made me run",
 "I fear my phone battery dying during an emergency call",
 "The injection at the hospital made me tremble with fear",
 "I am anxious that my flight might get cancelled tonight",
 "The fire alarm in the hostel woke everyone in panic",
 "I worry that my little brother is lost in the crowd",
 "The bridge swaying in the strong wind frightened everyone",
]
SURPRISE = [
 "Wow I never expected to win the lottery prize today",
 "My friends threw a surprise party I had no idea about",
 "I opened the gift box and found a brand new phone inside",
 "The magician made the coin disappear right before my eyes",
 "I suddenly saw my childhood friend after ten years",
 "The results came out and I topped the whole university",
 "Oh my god the fireworks started without any warning",
 "I found a hundred rupee note inside my old jacket pocket",
 "The teacher announced a holiday nobody was expecting",
 "My dog learned a new trick overnight and amazed us all",
 "The quiet student in class won the national debate",
 "I bumped into my favourite actor at the coffee shop",
 "The old house we visited had a hidden room behind the wall",
 "Suddenly it started snowing in the middle of April",
 "My grandmother cooked my favourite dish without telling me",
 "The train arrived an hour early which never happens",
 "I discovered a secret garden behind our college building",
 "The baby said her first word out of nowhere today",
 "My brother returned home from abroad without any notice",
 "The street performer pulled a rabbit out of his hat",
 "I found my lost wallet in the pocket of my other jeans",
 "The rain stopped and a double rainbow appeared in the sky",
 "My favourite singer announced a surprise concert tonight",
 "The old clock that stopped years ago suddenly started ticking",
 "I met my school teacher in a different country by chance",
 "The book I was reading had a shocking twist ending",
 "A butterfly landed on my hand while I was sitting in the park",
 "The quiet library suddenly echoed with a loud celebration",
 "I received a letter from a friend I had not heard from in years",
 "The match ended in a draw after an unbelievable last ball six",
]
LOVE = [
 "I love spending quiet evenings with my family at home",
 "Her smile makes my whole day brighter every single time",
 "I adore the way my mother sings while cooking in the kitchen",
 "My heart melts when my little sister hugs me tight",
 "I cherish every moment I spend with my best friends",
 "He always remembers the small things that make me happy",
 "I love the smell of rain on the first day of monsoon",
 "Holding hands with someone you love feels like home",
 "I am grateful for my parents who sacrificed so much for me",
 "My grandmother's stories warm my heart even today",
 "I love watching the stars with the person I care about",
 "She waited for me at the station with my favourite snacks",
 "I adore my dog who greets me with joy every morning",
 "The love in my family keeps me strong through hard times",
 "I cherish the letters my friend wrote to me years ago",
 "He surprised me with flowers just because it was Tuesday",
 "I love the feeling of coming home after a long journey",
 "My heart is full when I see my parents smile proudly",
 "I adore listening to old songs with my grandfather",
 "She held my hand when I was scared and I felt safe",
 "I love my hometown with all its dusty roads and memories",
 "The warmth of a hug from my mother heals everything",
 "I cherish our late night talks under the hostel stars",
 "He cooks my favourite meal whenever I feel low",
 "I love the way my friends laugh at my silly jokes",
 "My sister and I share secrets nobody else will ever know",
 "I am thankful for the people who stood by me always",
 "Holding my newborn niece for the first time filled me with love",
 "I adore the quiet mornings we spend together drinking chai",
 "Her kindness towards strangers inspires me every day",
]


from extra_data import JOY_EXTRA, SADNESS_EXTRA, ANGER_EXTRA, FEAR_EXTRA, SURPRISE_EXTRA, LOVE_EXTRA
JOY += JOY_EXTRA; SADNESS += SADNESS_EXTRA; ANGER += ANGER_EXTRA
FEAR += FEAR_EXTRA; SURPRISE += SURPRISE_EXTRA; LOVE += LOVE_EXTRA

CLASSES = ["joy", "sadness", "anger", "fear", "surprise", "love"]
DATA = [JOY, SADNESS, ANGER, FEAR, SURPRISE, LOVE]
texts = [s for group in DATA for s in group]
y = np.array([i for i, group in enumerate(DATA) for _ in group])

vec = TfidfVectorizer(sublinear_tf=True, lowercase=True)
X = vec.fit_transform(texts)
model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
pred_cv = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), X, y, cv=skf)
acc = float(accuracy_score(y, pred_cv))
base = float(max(np.bincount(y)) / len(y))
print(f"n={len(texts)} per_class={[len(g) for g in DATA]} vocab={len(vec.vocabulary_)}")
print(f"CV accuracy={acc:.3f} baseline={base:.3f}")
print(classification_report(y, pred_cv, target_names=CLASSES, digits=3))

mj = {"classes": CLASSES, "vocabulary": vec.vocabulary_, "idf": vec.idf_.tolist(),
      "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
      "metrics": {"cv_accuracy": acc, "baseline_accuracy": base,
                  "n": len(texts), "per_class": [len(g) for g in DATA],
                  "vocab": len(vec.vocabulary_)}}
with open("model.json", "w") as f:
    json.dump(mj, f, indent=1)

def py_probs(text):
    v = vec.transform([text])
    return model.predict_proba(v)[0].tolist()

samples = [
 "I just got the internship I have been dreaming about for months",
 "I miss my grandmother who passed away last year",
 "Someone stole my bicycle from outside the college gate",
 "I heard strange noises in the empty house last night",
 "Wow I never expected to win the lottery prize today",
 "Her smile makes my whole day brighter every single time",
 "The quarterly report was submitted on Tuesday afternoon",
]
for s in samples:
    p = py_probs(s)
    print(repr(s), "->", CLASSES[int(np.argmax(p))], [round(x*100, 2) for x in p])
with open("parity_samples.json", "w") as f:
    json.dump([{"text": s, "probs": py_probs(s)} for s in samples], f, indent=1)
