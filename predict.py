# Import libraries
import joblib
import numpy as np

def make_prediction(text, pipeline_path):
    # Loading pipeline
    pipeline = joblib.load(pipeline_path)
    vectorizer = pipeline['vectorizer']
    classifier = pipeline['classifier']

    # Adjusting inputs format
    if isinstance(text, str):
        text = [text]

    preprocessed_text = vectorizer.transform(text)
    classifications = classifier.predict(preprocessed_text)
    return classifications

if __name__ == '__main__':
    reviews = [
        "I went in with low expectations and came out pleasantly surprised. The lead gave such a quiet, honest performance that I kept thinking about one scene for days. It’s not flashy, but it stuck with me — highly recommend if you like character-driven stories.",
        "Saw this with my partner and we both laughed out loud a few times — the chemistry between the two leads felt real. A couple of plot threads fizzle out, but the moments that work are genuinely warm and funny. Left the theater smiling.",
        "Really clever script and a few jokes that actually surprised me. The pacing slows a bit in the middle, but the final 20 minutes paid off. I enjoyed how the director trusted the audience and didn’t spell everything out.",
        "A small, intimate film that felt honest and lived-in. The acting is raw (in a good way) and the ending stuck with me — not tied up, but emotionally true. It’s the kind of movie I’d recommend to friends who like slow-burn dramas.",
        "Totally crowd-pleasing — my group had a blast. It’s loud and a little predictable, but the set pieces are fun, the soundtrack is catchy, and it never drags. Perfect for a weekend night out.",
        "Such a frustrating watch. The premise had me interested, but scenes kept going nowhere and characters made choices that made no sense. I wanted to like it, but by the second act I was checking my phone more than paying attention.",
        "Too long and mostly boring. A few jokes land, but most of the dialogue felt clunky and forced. I fell asleep for a short stretch and missed what I guess was the big reveal — not a good sign.",
        "This felt like a checklist of drama clichés. The performances are fine but the script gives them nothing original to do. The ending was exactly what I expected — predictable and kind of unsatisfying.",
        "Beautiful shots, terrible movie. The director keeps showing the same visual trick over and over as if that replaces character development. I wanted an emotional connection and never got one.",
        "Loved the trailer, hated the movie. It feels padded with filler scenes and the villain's motives are laughable. A few moments work, but overall it wasted a good cast and an okay idea."
    ]

    print(make_prediction(reviews,'models/tfidf_svm_review_polarity.joblib'), "Model trained on review polarity dataset")
    print(make_prediction(reviews, 'models/tfidf_svm_rt-polaritydata.joblib'), "Model trained on rt-polarity dataset")
    print(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), "Correct labels")