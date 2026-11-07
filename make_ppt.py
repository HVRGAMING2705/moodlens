from pptx import Presentation
from pptx.util import Inches, Pt
import sys
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "moodlens.vercel.app"
prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
def add(i, title, lines):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame; tf.word_wrap = True
    for j, ln in enumerate(lines):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = ln; p.font.size = Pt(20); p.space_after = Pt(10)
add(0, "MoodLens — Emotion Detector from Text", ["An AI/ML project by Vikranth Reddy", "B.Tech CSE (AI & ML), 2nd Year — Woxsen University", "A model that reads a sentence and tells you which of six emotions it carries — joy, sadness, anger, fear, surprise or love"])
add(1, "The Problem", ["People write how they feel every day — in messages, reviews and posts", "Reading the emotion behind words takes time, and it is easy to misread tone online", "Businesses want to know how customers feel; counsellors want early signs of distress", "An automatic emotion check gives a fast first read of any sentence"])
add(2, "The Solution — MoodLens", ["Type or paste any sentence into the browser demo", "A trained model returns the probability of each of six emotions as bars", "The strongest emotion is shown with an emoji and a percentage", "Runs fully in the browser — no backend, no waiting"])
add(3, "Tech Stack", ["Python + scikit-learn: dataset and model training", "TF-IDF (word unigrams, sublinear TF, L2 normalisation) + multinomial logistic regression (C = 1.0)", "HTML, CSS and vanilla JavaScript for the interactive demo", "GitHub for source control, Vercel for free hosting"])
add(4, "Architecture", ["Dataset: 300 hand-written sentences — 50 per emotion, 796-word vocabulary", "Text -> tokens -> TF-IDF weights -> logistic regression (6 outputs) -> softmax -> model.json", "model.json ships with the site; JavaScript applies the same formula and a softmax", "Training script: train.py; site generator: gen_site.py"])
add(5, "Live Demo", ["Try it: type a sentence and press Detect Emotion", "Joy example: I just got the internship I have been dreaming about for months -> joy, about 38.5%", "Sadness example: I miss my grandmother who passed away last year -> sadness, about 44.0%", f"Demo: https://{DOMAIN}"])
add(6, "Results", ["5-fold cross-validation accuracy: 72.0%", "Baseline (always guessing the most common emotion): 16.7% accuracy — a 4.3x lift", "Best classes: love F1 0.83, fear F1 0.82, anger F1 0.79; hardest: joy F1 0.56", "Browser demo matches Python probabilities exactly on all test samples"])
add(7, "Future Scope", ["Train on a larger real-world emotion dataset with thousands of examples", "Add intensity levels — slightly sad vs deeply sad", "Support more languages, starting with Telugu and Hindi", "Add a chatbot that replies with empathy based on the detected emotion"])
add(8, "Thank You", ["MoodLens — Emotion Detector from Text", "Vikranth Reddy, B.Tech CSE (AI & ML), Woxsen University", "Source: github.com/HVRGAMING2705/moodlens", "Questions welcome!"])
prs.save("/home/hatch/workspace/goals/daily-ai-ml-project-factory/files/AIML_MoodLens_2026-11-07.pptx")
print("slides:", len(prs.slides._sldIdLst))
